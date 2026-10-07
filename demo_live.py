"""Full pipeline demo: video -> YOLOv8 counts -> trained DQN -> recommended green phase.
Usage: python demo_live.py [videos/traffic.mp4 | 0 for webcam] [--interval 5]
The agent decides once every --interval seconds of video (like the 5 s step used in training).
"""
import argparse
from collections import deque

import cv2
import numpy as np
from stable_baselines3 import DQN

from yolo_counter import count_zones
from sumo_env import MAX_COUNT, MAX_PHASES

ap = argparse.ArgumentParser()
ap.add_argument("source", nargs="?", default="videos/traffic.mp4")
ap.add_argument("--interval", type=float, default=5.0)
ap.add_argument("--scale", type=float, default=1.0,
                help="multiply video counts to match SUMO density (tune if your video is sparse/dense)")
args = ap.parse_args()

# Must match the phase order printed by sumo_env.py (default network: phase 0 = N+S, phase 1 = E+W)
PHASE_LABELS = ["N+S", "E+W"]

model = DQN.load("models/dqn_traffic")
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
every = max(1, int(fps * args.interval))
hist, cur, frame_no = deque(maxlen=int(fps)), 0, 0   # ~1 s smoothing window

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    counts, frame = count_zones(frame)
    hist.append([counts[z] for z in "NESW"])
    if frame_no % every == 0 and frame_no > 0:
        avg = np.mean(hist, axis=0) * args.scale
        onehot = np.zeros(MAX_PHASES)
        onehot[cur] = 1
        obs = np.array(list(np.minimum(avg / MAX_COUNT, 1.0)) + list(onehot), dtype=np.float32)
        cur = int(model.predict(obs, deterministic=True)[0])
    frame_no += 1
    cv2.putText(frame, f"AI decision: GREEN for {PHASE_LABELS[cur]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    cv2.imshow("Adaptive signal demo (q to quit)", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

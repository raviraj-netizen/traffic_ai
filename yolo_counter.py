"""YOLOv8 vehicle counting per approach zone.   Test it alone:  python yolo_counter.py videos/traffic.mp4"""
import json
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

VEHICLES = {2, 3, 5, 7}                      # COCO ids: car, motorcycle, bus, truck
HERE = os.path.dirname(os.path.abspath(__file__))
ZONES = {k: np.array(v, dtype=np.int32)
         for k, v in json.load(open(os.path.join(HERE, "zones.json"))).items()}
model = YOLO("yolov8n.pt")                   # auto-downloads on first run (use yolov8s.pt for accuracy)


def count_zones(frame, conf=0.3):
    """Returns ({'N':n,'E':n,'S':n,'W':n}, annotated_frame)."""
    res = model(frame, verbose=False, conf=conf)[0]
    counts = {z: 0 for z in ZONES}
    for box, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
        if int(cls) not in VEHICLES:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for z, poly in ZONES.items():
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                counts[z] += 1
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    for z, poly in ZONES.items():
        cv2.polylines(frame, [poly], True, (255, 0, 0), 2)
        cv2.putText(frame, f"{z}:{counts[z]}", tuple(int(v) for v in poly[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return counts, frame


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "videos/traffic.mp4"
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        _, frame = count_zones(frame)
        cv2.imshow("YOLO counts (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

"""Draw the 4 approach zones on your video frame by clicking, and save them to zones.json.
Usage: python calibrate_zones.py videos/traffic.mp4
For each approach in order N, E, S, W: click 4 corners of the road area, (the lane cars wait in).
Keys: u = undo last point, q = quit without saving.
"""
import json
import sys
import cv2
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "videos/traffic.mp4"
cap = cv2.VideoCapture(src)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"Could not read a frame from {src}")

NAMES, pts = ["N", "E", "S", "W"], []


def click(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 16:
        pts.append([x, y])


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", click)
while True:
    img = frame.copy()
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
    for i in range(len(pts) // 4):
        cv2.polylines(img, [np.array(pts[i * 4:i * 4 + 4])], True, (255, 0, 0), 2)
        cv2.putText(img, NAMES[i], tuple(pts[i * 4]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    nxt = NAMES[len(pts) // 4] if len(pts) < 16 else "done - press s to save"
    cv2.putText(img, f"Click 4 corners for: {nxt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("calibrate", img)
    k = cv2.waitKey(30) & 0xFF
    if k == ord("u") and pts:
        pts.pop()
    elif k == ord("q"):
        break
    elif k == ord("s") and len(pts) == 16:
        json.dump({n: pts[i * 4:i * 4 + 4] for i, n in enumerate(NAMES)}, open("zones.json", "w"), indent=2)
        print("Saved zones.json")
        break
cv2.destroyAllWindows()

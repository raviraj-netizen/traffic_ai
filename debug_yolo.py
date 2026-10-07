import sys, cv2
from ultralytics import YOLO

src = sys.argv[1] if len(sys.argv) > 1 else "videos/traffic.mp4"
model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(src)
ok, frame = cap.read()
print("Frame size (width x height):", frame.shape[1], "x", frame.shape[0])

res = model(frame, conf=0.15, imgsz=1280, verbose=False)[0]
print("Total detections:", len(res.boxes))
for c, p in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
    print(f"  {model.names[int(c)]:12s} confidence {p:.2f}")

cv2.imshow("all detections (any key to close)", res.plot())
cv2.waitKey(0)
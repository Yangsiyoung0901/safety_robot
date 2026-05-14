from ultralytics import YOLO
m = YOLO(r"D:\AIProject\YSY\best.pt")
r = m.predict(r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\large\images\a02087.jpg", conf=0.05, verbose=False)[0]  # conf 아주 낮게
for b in r.boxes:
    print(m.names[int(b.cls[0])], round(float(b.conf[0]), 3))
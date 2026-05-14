import argparse
import html
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent

# Put your trained PPE model here. The first existing file is used.
PPE_MODEL_CANDIDATES = [
    APP_DIR / "best.pt",
    APP_DIR / "best_mixed.pt",
    Path("/home/ryu/project/software/best.pt"),
    Path("/home/ryu/project/software/best_mixed.pt"),
]

# Person detector is used to number people. If it is missing, the web stream still runs.
PERSON_MODEL_CANDIDATES = [
    APP_DIR / "yolo11n.pt",
    APP_DIR / "yolov8n.pt",
    Path("/home/ryu/project/software/yolo11n.pt"),
    Path("/home/ryu/project/software/yolov8n.pt"),
]

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_SIZE = 320
PERSON_CONF = 0.34
PPE_CONF = 0.35
PERSON_BOX_PADDING = 0.08
PANEL_WIDTH = 340
JPEG_QUALITY = 68

# Speed knobs. Larger values are faster but person numbering updates less often.
PERSON_DETECT_EVERY = 3
MAX_STREAM_FPS = 12

# Brightness correction. If the image is still too bright, lower BRIGHTNESS_BETA.
BRIGHTNESS_ALPHA = 1.00
BRIGHTNESS_BETA = 0
GAMMA = 1.1

# Camera controls. Some USB cameras ignore these.
CAMERA_BRIGHTNESS = 70
CAMERA_CONTRAST = 25
CAMERA_AUTO_EXPOSURE = 3
CAMERA_EXPOSURE = -4


latest_jpeg = None
latest_status = "Starting"
latest_lock = threading.Lock()
stop_event = threading.Event()


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def find_class_ids(model, target_names):
    target_norms = {normalize_name(name) for name in target_names}
    return {
        cls_id
        for cls_id, name in model.names.items()
        if normalize_name(name) in target_norms
    }


def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def expand_box(box, frame_shape, padding_ratio):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding_ratio
    pad_y = bh * padding_ratio
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w - 1, x2 + pad_x),
        min(h - 1, y2 + pad_y),
    )


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def status_color(has_helmet, has_vest):
    if has_helmet and has_vest:
        return (0, 220, 0)
    if not has_helmet and not has_vest:
        return (0, 0, 255)
    return (0, 165, 255)


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(
        img,
        str(text),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def adjust_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)
    if GAMMA != 1.0:
        table = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)
    return frame


def make_panel(height, persons, no_helmet_nums, no_vest_nums, fps, ppe_count):
    panel = np.full((height, PANEL_WIDTH, 3), (28, 28, 28), dtype=np.uint8)

    draw_text(panel, "PPE STATUS", (22, 46), (255, 255, 255), 0.95, 2)
    draw_text(panel, f"People: {len(persons)}", (22, 92), (220, 220, 220), 0.68, 2)
    draw_text(panel, f"PPE boxes: {ppe_count}", (22, 124), (220, 220, 220), 0.62, 2)
    draw_text(panel, f"FPS: {fps:.1f}", (22, 156), (180, 220, 255), 0.62, 2)

    helmet_text = ", ".join(str(num) for num in no_helmet_nums) if no_helmet_nums else "None"
    vest_text = ", ".join(str(num) for num in no_vest_nums) if no_vest_nums else "None"

    draw_text(panel, "No Helmet", (22, 220), (180, 180, 255), 0.72, 2)
    draw_text(panel, helmet_text, (22, 262), (0, 0, 255), 0.9, 2)

    draw_text(panel, "No Vest", (22, 330), (180, 220, 255), 0.72, 2)
    draw_text(panel, vest_text, (22, 372), (0, 165, 255), 0.9, 2)

    y = 450
    draw_text(panel, "Person Results", (22, y), (255, 255, 255), 0.68, 2)
    y += 38

    for person in persons[:8]:
        num = person["num"]
        has_helmet = person["has_helmet"]
        has_vest = person["has_vest"]

        if has_helmet and has_vest:
            status = "SAFE"
            color = (0, 220, 0)
        else:
            missing = []
            if not has_helmet:
                missing.append("NO HELMET")
            if not has_vest:
                missing.append("NO VEST")
            status = " / ".join(missing)
            color = status_color(has_helmet, has_vest)

        draw_text(panel, f"#{num}: {status}", (22, y), color, 0.55, 2)
        y += 32

    return panel


def update_latest(jpeg_bytes, status):
    global latest_jpeg, latest_status
    with latest_lock:
        latest_jpeg = jpeg_bytes
        latest_status = status


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Restore camera auto exposure.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    time.sleep(1.0)

    return cap


def detection_loop():
    ppe_model_path = first_existing(PPE_MODEL_CANDIDATES)
    if ppe_model_path is None:
        update_latest(None, "PPE model not found")
        print("PPE model not found. Put best.pt in /home/ryu/project/software.")
        return

    person_model_path = first_existing(PERSON_MODEL_CANDIDATES)

    print("Loading PPE model:", ppe_model_path)
    ppe_model = YOLO(str(ppe_model_path))
    person_model = None
    if person_model_path is not None:
        print("Loading person model:", person_model_path)
        person_model = YOLO(str(person_model_path))
    else:
        print("Person model not found. Person numbering will be limited.")

    helmet_class_ids = find_class_ids(ppe_model, {"helmet", "hardhat", "hard_hat", "safety_helmet"})
    vest_class_ids = find_class_ids(ppe_model, {"vest", "safety_vest", "safetyvest"})
    head_class_ids = find_class_ids(ppe_model, {"head", "person"})
    person_class_ids = find_class_ids(person_model, {"person"}) if person_model is not None else set()
    if person_model is not None and not person_class_ids:
        person_class_ids = {0}

    print("PPE model classes:", ppe_model.names)
    print("helmet ids:", helmet_class_ids, "vest ids:", vest_class_ids, "head ids:", head_class_ids)

    cap = open_camera()
    if not cap.isOpened():
        update_latest(None, "Could not open camera")
        print("Could not open camera.")
        return

    cached_persons = []
    frame_index = 0
    fps = 0.0
    last_time = time.monotonic()
    min_frame_interval = 1.0 / max(MAX_STREAM_FPS, 1)

    print("Camera web monitor started.")
    while not stop_event.is_set():
        loop_start = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            update_latest(None, "Could not read camera frame")
            time.sleep(0.1)
            continue

        frame = adjust_frame(frame)

        should_update_persons = (
            person_model is not None
            and (frame_index % max(PERSON_DETECT_EVERY, 1) == 0 or not cached_persons)
        )

        if should_update_persons:
            person_result = person_model.predict(
                source=frame,
                imgsz=YOLO_SIZE,
                conf=PERSON_CONF,
                verbose=False,
            )[0]
            cached_persons = []
            for box in person_result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in person_class_ids:
                    continue
                xyxy = tuple(float(v) for v in box.xyxy[0])
                cached_persons.append({"box": xyxy, "conf": float(box.conf[0])})
            cached_persons.sort(key=lambda item: item["box"][0])

        ppe_result = ppe_model.predict(
            source=frame,
            imgsz=YOLO_SIZE,
            conf=PPE_CONF,
            verbose=False,
        )[0]

        helmets = []
        vests = []
        heads = []
        for box in ppe_result.boxes:
            cls_id = int(box.cls[0])
            xyxy = tuple(float(v) for v in box.xyxy[0])
            item = {
                "box": xyxy,
                "conf": float(box.conf[0]),
                "center": center_of(xyxy),
            }

            if cls_id in helmet_class_ids:
                helmets.append(item)
            elif cls_id in vest_class_ids:
                vests.append(item)
            elif cls_id in head_class_ids:
                heads.append(item)

        no_helmet_nums = []
        no_vest_nums = []
        numbered_persons = []

        for idx, person in enumerate(cached_persons, start=1):
            person_box = person["box"]
            match_box = expand_box(person_box, frame.shape, PERSON_BOX_PADDING)

            has_helmet = any(point_in_box(item["center"], match_box) for item in helmets)
            has_vest = any(point_in_box(item["center"], match_box) for item in vests)

            if not has_helmet:
                no_helmet_nums.append(idx)
            if not has_vest:
                no_vest_nums.append(idx)

            numbered_persons.append({
                "num": idx,
                "box": person_box,
                "has_helmet": has_helmet,
                "has_vest": has_vest,
            })

            x1, y1, x2, y2 = [int(v) for v in person_box]
            color = status_color(has_helmet, has_vest)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x1 + 22, y1 + 22), 20, color, -1)
            draw_text(frame, str(idx), (x1 + 11, y1 + 32), (255, 255, 255), 0.9, 2)

        # Draw PPE detections as thin guide boxes.
        for item in helmets:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for item in vests:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 1)
        for item in heads:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 1)

        now = time.monotonic()
        elapsed = max(now - last_time, 0.001)
        last_time = now
        fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

        ppe_count = len(helmets) + len(vests) + len(heads)
        panel = make_panel(frame.shape[0], numbered_persons, no_helmet_nums, no_vest_nums, fps, ppe_count)
        output = np.hstack((frame, panel))
        ok, jpeg = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            update_latest(jpeg.tobytes(), f"OK people={len(numbered_persons)} ppe={ppe_count} fps={fps:.1f}")

        frame_index += 1
        spent = time.monotonic() - loop_start
        if spent < min_frame_interval:
            time.sleep(min_frame_interval - spent)

    cap.release()


class MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_index()
        elif self.path == "/stream.mjpg":
            self.send_stream()
        elif self.path == "/status":
            self.send_status()
        else:
            self.send_error(404, "Not found")

    def send_index(self):
        page = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safe Eye PPE Monitor</title>
  <style>
    body { margin: 0; background: #101010; color: white; font-family: Arial, sans-serif; }
    header { padding: 12px 18px; background: #1e1e1e; font-size: 20px; font-weight: 700; }
    img { display: block; width: 100vw; height: calc(100vh - 50px); object-fit: contain; background: #000; }
  </style>
</head>
<body>
  <header>Safe Eye PPE Monitor</header>
  <img src="/stream.mjpg" alt="PPE camera stream">
</body>
</html>
"""
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_status(self):
        with latest_lock:
            status = latest_status
        body = html.escape(status).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while not stop_event.is_set():
            with latest_lock:
                frame = latest_jpeg
            if frame is None:
                time.sleep(0.1)
                continue

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                break


def main():
    parser = argparse.ArgumentParser(description="Safe Eye web PPE monitor for Raspberry Pi.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    worker = threading.Thread(target=detection_loop, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"Open from PC: http://<raspberry-pi-ip>:{args.port}")
    print("Example: http://10.10.141.134:8000")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()

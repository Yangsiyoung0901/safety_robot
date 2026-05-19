import argparse
import html
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("Warning: ultralytics package not available — model inference disabled")


APP_DIR = Path(__file__).resolve().parent

PPE_MODEL_CANDIDATES = [
    APP_DIR / "best_p.pt",
    APP_DIR / "best_mixed.pt",
    Path("/home/ryu/project/software/best.pt"),
    Path("/home/ryu/project/software/best_mixed.pt"),
]

PERSON_MODEL_CANDIDATES = [
    APP_DIR / "yolo11n.pt",
    APP_DIR / "yolov8n.pt",
    Path("/home/ryu/project/software/yolo11n.pt"),
    Path("/home/ryu/project/software/yolov8n.pt"),
]

# Put the trained danger sign model here after training.
DANGER_SIGN_MODEL_CANDIDATES = [
    APP_DIR / "best_p.pt",
    APP_DIR / "danger_sign_best.pt",
    APP_DIR / "best_danger_sign.pt",
    APP_DIR / "runs" / "detect" / "danger_sign_yolo11n" / "weights" / "best.pt",
    Path("/home/ryu/project/software/danger_sign_yolo11n.pt"),
    Path("/home/ryu/project/software/danger_sign_best.pt"),
]

CAMERA_INDEX = 0
# Increase resolution; ensure your camera supports this (e.g., 1280x720)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
YOLO_SIZE = 320
PERSON_CONF = 0.34
PPE_CONF = 0.35
SIGN_CONF = 0.45
SIGN_YOLO_SIZE = 416
PERSON_BOX_PADDING = 0.08
PANEL_WIDTH = 360
JPEG_QUALITY = 95

# Speed knobs.
PERSON_DETECT_EVERY = 3
MAX_STREAM_FPS = 12

# Danger sign detection is intentionally sparse. The sign is fixed, so cache it.
SIGN_DETECT_WARMUP_FRAMES = 20
SIGN_REFRESH_SECONDS = 10.0
DANGER_ZONE_SCALE = 3.0
DANGER_HOLD_FRAMES = 3

# If the camera is fixed and you want zero sign-model cost, add manual zones here.
# Format: (x1, y1, x2, y2) in camera frame pixels.
MANUAL_DANGER_ZONES = [
    # (420, 120, 650, 420),
]

BRIGHTNESS_ALPHA = 1.00
BRIGHTNESS_BETA = 0
GAMMA = 1.1

CAMERA_BRIGHTNESS = 80
CAMERA_CONTRAST = 35
CAMERA_AUTO_EXPOSURE = 3
CAMERA_EXPOSURE = -4


latest_jpeg = None
latest_status = "Starting"
latest_data = {}
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
    if model is None:
        return set()
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


def expand_zone_from_sign(sign_box, frame_shape, scale):
    x1, y1, x2, y2 = sign_box
    h, w = frame_shape[:2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = max(1.0, x2 - x1) * scale
    bh = max(1.0, y2 - y1) * scale
    return (
        max(0, cx - bw / 2),
        max(0, cy - bh / 2),
        min(w - 1, cx + bw / 2),
        min(h - 1, cy + bh / 2),
    )


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


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
    cv2.putText(img, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def adjust_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)
    if GAMMA != 1.0:
        table = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)
    return frame


def make_panel(height, persons, no_helmet_nums, no_vest_nums, danger_nums, fps, ppe_count, sign_count, danger_active):
    panel = np.full((height, PANEL_WIDTH, 3), (28, 28, 28), dtype=np.uint8)

    draw_text(panel, "SAFE EYE STATUS", (22, 42), (255, 255, 255), 0.85, 2)
    draw_text(panel, f"People: {len(persons)}", (22, 84), (220, 220, 220), 0.62, 2)
    draw_text(panel, f"PPE boxes: {ppe_count}", (22, 114), (220, 220, 220), 0.58, 2)
    draw_text(panel, f"Danger signs: {sign_count}", (22, 144), (220, 220, 220), 0.58, 2)
    draw_text(panel, f"FPS: {fps:.1f}", (22, 174), (180, 220, 255), 0.58, 2)

    danger_text = ", ".join(str(num) for num in danger_nums) if danger_nums else "None"
    danger_color = (0, 0, 255) if danger_active else (0, 180, 0)
    draw_text(panel, "Danger Zone", (22, 226), (255, 210, 180), 0.70, 2)
    draw_text(panel, danger_text, (22, 268), danger_color, 0.9, 2)

    helmet_text = ", ".join(str(num) for num in no_helmet_nums) if no_helmet_nums else "None"
    vest_text = ", ".join(str(num) for num in no_vest_nums) if no_vest_nums else "None"

    draw_text(panel, "No Helmet", (22, 330), (180, 180, 255), 0.66, 2)
    draw_text(panel, helmet_text, (22, 366), (0, 0, 255), 0.78, 2)

    draw_text(panel, "No Vest", (22, 420), (180, 220, 255), 0.66, 2)
    draw_text(panel, vest_text, (22, 456), (0, 165, 255), 0.78, 2)

    y = 515
    draw_text(panel, "Person Results", (22, y), (255, 255, 255), 0.62, 2)
    y += 32

    for person in persons[:6]:
        num = person["num"]
        has_helmet = person["has_helmet"]
        has_vest = person["has_vest"]
        in_danger = person["in_danger"]

        if in_danger:
            status = "DANGER ZONE"
            color = (0, 0, 255)
        elif has_helmet and has_vest:
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

        draw_text(panel, f"#{num}: {status}", (22, y), color, 0.50, 2)
        y += 29

    return panel


def update_latest(jpeg_bytes, status):
    global latest_jpeg, latest_status, latest_data
    with latest_lock:
        latest_jpeg = jpeg_bytes
        # status may be a dict (structured JSON) or a legacy text string
        if isinstance(status, dict):
            latest_data = status
            latest_status = "OK"
        else:
            latest_status = status
            try:
                latest_data = {"status": status}
            except Exception:
                latest_data = {}


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAMERA_AUTO_EXPOSURE)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS)
    cap.set(cv2.CAP_PROP_CONTRAST, CAMERA_CONTRAST)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)
    time.sleep(1.0)
    return cap


def detect_signs(sign_model, sign_class_ids, frame):
    if sign_model is None:
        return []
    result = sign_model.predict(source=frame, imgsz=SIGN_YOLO_SIZE, conf=SIGN_CONF, verbose=False)[0]
    signs = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if sign_class_ids and cls_id not in sign_class_ids:
            continue
        xyxy = tuple(float(v) for v in box.xyxy[0])
        signs.append({"box": xyxy, "conf": float(box.conf[0])})
    return signs


def detection_loop():
    ppe_model_path = first_existing(PPE_MODEL_CANDIDATES)
    if ppe_model_path is None:
        update_latest(None, "PPE model not found")
        print("PPE model not found. Continuing without PPE model — camera stream only.")
        ppe_model_path = None

    person_model_path = first_existing(PERSON_MODEL_CANDIDATES)
    sign_model_path = first_existing(DANGER_SIGN_MODEL_CANDIDATES)

    print("PPE model path:", ppe_model_path)
    if YOLO is None:
        print("ultralytics not available — skipping model initialisation")
        ppe_model = None
    else:
        print("Loading PPE model:", ppe_model_path)
        ppe_model = YOLO(str(ppe_model_path))

    person_model = None
    if person_model_path is not None and YOLO is not None:
        print("Loading person model:", person_model_path)
        person_model = YOLO(str(person_model_path))
    elif person_model_path is None:
        print("Person model not found. Person numbering will be limited.")
    else:
        print("Person model path found but ultralytics missing; skipping person model")

    sign_model = None
    if sign_model_path is not None and YOLO is not None:
        print("Loading danger sign model:", sign_model_path)
        sign_model = YOLO(str(sign_model_path))
    elif sign_model_path is None:
        print("Danger sign model not found. Only MANUAL_DANGER_ZONES will be used.")
    else:
        print("Danger sign model path found but ultralytics missing; skipping sign model")

    helmet_class_ids = find_class_ids(ppe_model, {"helmet", "hardhat", "hard_hat", "safety_helmet"})
    vest_class_ids = find_class_ids(ppe_model, {"vest", "safety_vest", "safetyvest"})
    head_class_ids = find_class_ids(ppe_model, {"head", "person"})
    person_class_ids = find_class_ids(person_model, {"person"}) if person_model is not None else set()
    sign_class_ids = find_class_ids(sign_model, {"danger_sign", "warning_sign", "sign"}) if sign_model is not None else set()

    if person_model is not None and not person_class_ids:
        person_class_ids = {0}
    if sign_model is not None and not sign_class_ids:
        sign_class_ids = {0}

    if ppe_model is None:
        print("PPE model not loaded, running camera-only stream.")
        cap = open_camera()
        if not cap.isOpened():
            update_latest(None, "Could not open camera")
            print("Could not open camera.")
            return

        min_frame_interval = 1.0 / max(MAX_STREAM_FPS, 1)
        fps = 0.0
        last_time = time.monotonic()
        print("Camera-only stream started (no inference).")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                update_latest(None, "Could not read camera frame")
                time.sleep(0.1)
                continue

            frame = adjust_frame(frame)
            now = time.monotonic()
            elapsed = max(now - last_time, 0.001)
            last_time = now
            fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                data = {"people": 0, "ppe": 0, "signs": 0, "danger": False, "fps": float(f"{fps:.1f}"), "persons": []}
                update_latest(jpeg.tobytes(), data)

            spent = time.monotonic() - now
            if spent < min_frame_interval:
                time.sleep(min_frame_interval - spent)

        cap.release()
        return
    print("PPE model classes:", ppe_model.names)
    print("helmet ids:", helmet_class_ids, "vest ids:", vest_class_ids, "head ids:", head_class_ids)
    if sign_model is not None:
        print("Danger sign model classes:", sign_model.names, "sign ids:", sign_class_ids)

    cap = open_camera()
    if not cap.isOpened():
        update_latest(None, "Could not open camera")
        print("Could not open camera.")
        return

    cached_persons = []
    cached_signs = []
    frame_index = 0
    fps = 0.0
    danger_hold = 0
    last_time = time.monotonic()
    last_sign_refresh = 0.0
    min_frame_interval = 1.0 / max(MAX_STREAM_FPS, 1)

    print("Camera web monitor with danger-zone alert started.")
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
            person_result = person_model.predict(source=frame, imgsz=YOLO_SIZE, conf=PERSON_CONF, verbose=False)[0]
            cached_persons = []
            for box in person_result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in person_class_ids:
                    continue
                xyxy = tuple(float(v) for v in box.xyxy[0])
                cached_persons.append({"box": xyxy, "conf": float(box.conf[0])})
            cached_persons.sort(key=lambda item: item["box"][0])

        now = time.monotonic()
        should_refresh_signs = (
            sign_model is not None
            and (
                frame_index < SIGN_DETECT_WARMUP_FRAMES
                or not cached_signs
                or now - last_sign_refresh >= SIGN_REFRESH_SECONDS
            )
        )
        if should_refresh_signs:
            new_signs = detect_signs(sign_model, sign_class_ids, frame)
            if new_signs:
                cached_signs = new_signs
            last_sign_refresh = now

        ppe_result = ppe_model.predict(source=frame, imgsz=YOLO_SIZE, conf=PPE_CONF, verbose=False)[0]

        helmets = []
        vests = []
        heads = []
        for box in ppe_result.boxes:
            cls_id = int(box.cls[0])
            xyxy = tuple(float(v) for v in box.xyxy[0])
            item = {"box": xyxy, "conf": float(box.conf[0]), "center": center_of(xyxy)}

            if cls_id in helmet_class_ids:
                helmets.append(item)
            elif cls_id in vest_class_ids:
                vests.append(item)
            elif cls_id in head_class_ids:
                heads.append(item)

        danger_zones = [tuple(float(v) for v in zone) for zone in MANUAL_DANGER_ZONES]
        for sign in cached_signs:
            danger_zones.append(expand_zone_from_sign(sign["box"], frame.shape, DANGER_ZONE_SCALE))

        no_helmet_nums = []
        no_vest_nums = []
        danger_nums = []
        numbered_persons = []

        for idx, person in enumerate(cached_persons, start=1):
            person_box = person["box"]
            match_box = expand_box(person_box, frame.shape, PERSON_BOX_PADDING)

            has_helmet = any(point_in_box(item["center"], match_box) for item in helmets)
            has_vest = any(point_in_box(item["center"], match_box) for item in vests)
            in_danger = any(boxes_overlap(person_box, zone) for zone in danger_zones)

            if not has_helmet:
                no_helmet_nums.append(idx)
            if not has_vest:
                no_vest_nums.append(idx)
            if in_danger:
                danger_nums.append(idx)

            numbered_persons.append({
                "num": idx,
                "box": person_box,
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "in_danger": in_danger,
            })

            x1, y1, x2, y2 = [int(v) for v in person_box]
            color = (0, 0, 255) if in_danger else status_color(has_helmet, has_vest)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x1 + 22, y1 + 22), 20, color, -1)
            draw_text(frame, str(idx), (x1 + 11, y1 + 32), (255, 255, 255), 0.9, 2)
            if in_danger:
                draw_text(frame, "DANGER", (x1, max(25, y1 - 10)), (0, 0, 255), 0.75, 3)

        danger_hold = danger_hold + 1 if danger_nums else 0
        danger_active = danger_hold >= DANGER_HOLD_FRAMES

        for zone in danger_zones:
            x1, y1, x2, y2 = [int(v) for v in zone]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_text(frame, "DANGER ZONE", (x1, max(22, y1 - 8)), (0, 0, 255), 0.65, 2)

        for sign in cached_signs:
            x1, y1, x2, y2 = [int(v) for v in sign["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 255), 2)
            draw_text(frame, "SIGN", (x1, max(22, y1 - 8)), (0, 80, 255), 0.65, 2)

        if danger_active:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 54), (0, 0, 180), -1)
            draw_text(frame, "WARNING: PERSON NEAR DANGER SIGN", (18, 36), (255, 255, 255), 0.85, 2)

        for item in helmets:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for item in vests:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 1)
        for item in heads:
            x1, y1, x2, y2 = [int(v) for v in item["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 1)

        elapsed = max(now - last_time, 0.001)
        last_time = now
        fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

        ppe_count = len(helmets) + len(vests) + len(heads)
        # Encode camera frame only (remove hstack panel to reduce bandwidth)
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            data = {
                "people": len(numbered_persons),
                "ppe": ppe_count,
                "signs": len(cached_signs),
                "danger": bool(danger_active),
                "fps": float(f"{fps:.1f}"),
                "persons": [
                    {"num": p["num"], "has_helmet": p["has_helmet"], "has_vest": p["has_vest"], "in_danger": p["in_danger"]}
                    for p in numbered_persons
                ],
            }
            update_latest(jpeg.tobytes(), data)

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
                index_path = APP_DIR / "index.html"
                if index_path.exists():
                        body = index_path.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return

                # fallback
                page = """<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Safe Eye Danger Monitor</title>
    <style>
        body { margin: 0; background: #101010; color: white; font-family: Arial, sans-serif; }
        header { padding: 12px 18px; background: #1e1e1e; font-size: 20px; font-weight: 700; }
        img { display: block; width: 100vw; height: calc(100vh - 50px); object-fit: contain; background: #000; }
    </style>
</head>
<body>
    <header>Safe Eye PPE + Danger Sign Monitor</header>
    <img src="/stream.mjpg" alt="Safe Eye camera stream">
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
            data = latest_data.copy() if isinstance(latest_data, dict) else {"status": latest_status}
            status_text = latest_status

        # fallback: include status text if no structured data
        if not data:
            data = {"status": status_text}

        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
    parser = argparse.ArgumentParser(description="Safe Eye web PPE and danger-sign monitor.")
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

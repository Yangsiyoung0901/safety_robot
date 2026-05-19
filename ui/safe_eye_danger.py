"""
safe_eye_danger.py — Safe Eye 통합 웹 모니터 서버 (v2)

수정사항 (v2):
  1. PPE 탐지를 격프레임으로 실행 (CPU 부하 감소)
  2. 하이브리드 PPE: 1명→MBC 분류, 2명이상→YOLO OD
  3. 밝기 조절 개선 (CLAHE 적용)
  4. IR 센서 기본 활성화 — 감지 시에만 YOLO 실행

모듈 통합:
  - vision/camera.py      → LatestFrameCamera
  - Danger/danger_detector → DangerDetector
  - speaker/speaker.py    → DangerSpeaker
  - sensor/ir_sensor.py   → IRSensor
  - Main/detector.py      → PPEClassifier (MBC, 1명일 때)

실행:
  cd safe_eye
  python3 ui/safe_eye_danger.py --host 0.0.0.0 --port 8000
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# ultralytics 오프라인 모드
os.environ["YOLO_OFFLINE"] = "1"

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
    print("Warning: ultralytics package not available — model inference disabled")

# =============================================================
# 프로젝트 모듈 import
# =============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vision.camera import LatestFrameCamera
from Danger.danger_detector import DangerDetector
from speaker.speaker import DangerSpeaker

# MBC 분류기 (detector.py에서 가져옴 — TFLite/PyTorch 자동 선택)
try:
    from Main.detector import create_ppe_classifier, compute_upper_body_box, _is_head_only
    MBC_AVAILABLE = True
except Exception as e:
    MBC_AVAILABLE = False
    create_ppe_classifier = None
    print(f"Note: MBC classifier not available ({e})")

# IR 센서
try:
    from sensor.ir_sensor import IRSensor
    IR_AVAILABLE = True
except Exception:
    IRSensor = None
    IR_AVAILABLE = False
    print("Note: IR sensor module not available (RPi.GPIO required)")


# =============================================================
# 모델 경로 후보
# =============================================================

MODELS_DIR = PROJECT_DIR / "models"

PPE_MODEL_CANDIDATES = [
    MODELS_DIR / "best_p.pt",
    MODELS_DIR / "best.pt",
    APP_DIR / "best_p.pt",
]

PERSON_MODEL_CANDIDATES = [
    MODELS_DIR / "yolo11n.pt",
    MODELS_DIR / "yolov8n.pt",
    APP_DIR / "yolo11n.pt",
]

DANGER_SIGN_MODEL_CANDIDATES = [
    MODELS_DIR / "danger_sign_yolo11n.pt",
    MODELS_DIR / "signs_b.pt",
    MODELS_DIR / "best_b.pt",
    APP_DIR / "signs_b.pt",
]

# MBC 분류 모델 (.tflite 또는 .pth)
MBC_MODEL_CANDIDATES = [
    MODELS_DIR / "PPE_MobileNetV3Large_INT8.tflite",
    MODELS_DIR / "PPE_MobileNetV3Large.tflite",
    MODELS_DIR / "ppe_classifier.pth",
]

SPEAKER_ASSETS_DIR = PROJECT_DIR / "speaker" / "assets"

# =============================================================
# 설정값
# =============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
YOLO_SIZE = 320
PERSON_CONF = 0.34
PPE_CONF = 0.35
SIGN_CONF = 0.45
SIGN_YOLO_SIZE = 416
PERSON_BOX_PADDING = 0.08
JPEG_QUALITY = 95
MBC_THRESHOLD = 0.3    # MBC 분류 임계값

# ── [수정1] 속도 조절: PPE도 격프레임 ──
PERSON_DETECT_EVERY = 3   # person 감지 주기 (3프레임마다)
PPE_DETECT_EVERY = 3      # PPE 감지 주기 (3프레임마다, 기존: 매 프레임)
MAX_STREAM_FPS = 12

# 위험 구역 설정
DANGER_ZONE_SCALE = 3.0
DANGER_HOLD_FRAMES = 3
DANGER_ZONE_DISPLAY_SECONDS = 5.0
SIGN_REFRESH_SECONDS = 10.0

MANUAL_DANGER_ZONES = []

# ── [수정3] 밝기 조절: CLAHE 사용 ──
BRIGHTNESS_ALPHA = 1.0
BRIGHTNESS_BETA = 5
GAMMA = 1.0              # 1.0이면 감마 보정 안 함
USE_CLAHE = True          # CLAHE(적응형 히스토그램 균일화) 사용
CLAHE_CLIP = 2.0          # CLAHE 클리핑 한계
CLAHE_GRID = (8, 8)       # CLAHE 그리드 크기

# ── [수정4] IR 센서 기본 활성화 ──
IR_PIN = 17
IR_ENABLED = True         # ★ True: IR 감지 시에만 추론 실행

SPEAKER_COOLDOWN = 5.0

# =============================================================
# 전역 상태
# =============================================================

latest_jpeg = None
latest_status = "Starting"
latest_data = {}
latest_lock = threading.Lock()
stop_event = threading.Event()


# =============================================================
# 유틸 함수
# =============================================================

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
    """[수정3] 밝기 조절 — CLAHE 기반으로 개선"""
    # 기본 밝기/대비 보정
    if BRIGHTNESS_ALPHA != 1.0 or BRIGHTNESS_BETA != 0:
        frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_ALPHA, beta=BRIGHTNESS_BETA)

    # CLAHE: 어두운 곳은 밝게, 밝은 곳은 유지 — 조명 불균형 현장에 효과적
    if USE_CLAHE:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 감마 보정 (GAMMA != 1.0일 때만)
    if GAMMA != 1.0:
        table = np.array([((i / 255.0) ** GAMMA) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)

    return frame


def update_latest(jpeg_bytes, status):
    global latest_jpeg, latest_status, latest_data
    with latest_lock:
        latest_jpeg = jpeg_bytes
        if isinstance(status, dict):
            latest_data = status
            latest_status = "OK"
        else:
            latest_status = status
            latest_data = {"status": status}


# =============================================================
# 감지 메인 루프
# =============================================================

def detection_loop():
    # ── 모델 로드 ──
    ppe_model_path = first_existing(PPE_MODEL_CANDIDATES)
    person_model_path = first_existing(PERSON_MODEL_CANDIDATES)
    danger_sign_model_path = first_existing(DANGER_SIGN_MODEL_CANDIDATES)
    mbc_model_path = first_existing(MBC_MODEL_CANDIDATES)

    print(f"PPE OD model path: {ppe_model_path}")
    print(f"Person model path: {person_model_path}")
    print(f"Danger sign model path: {danger_sign_model_path}")
    print(f"MBC classifier path: {mbc_model_path}")

    if YOLO is None:
        print("ultralytics not available — skipping model initialisation")
        ppe_model = None
        person_model = None
    else:
        ppe_model = YOLO(str(ppe_model_path)) if ppe_model_path else None
        person_model = YOLO(str(person_model_path)) if person_model_path else None

    if ppe_model:
        print("PPE OD model classes:", ppe_model.names)
    if person_model:
        print("Person model loaded")

    # 클래스 ID 매핑
    helmet_class_ids = find_class_ids(ppe_model, {"helmet", "hardhat", "hard_hat", "safety_helmet"})
    vest_class_ids = find_class_ids(ppe_model, {"vest", "safety_vest", "safetyvest"})
    ppe_class_ids = sorted(helmet_class_ids | vest_class_ids)
    person_class_ids = find_class_ids(person_model, {"person"}) if person_model else set()

    if person_model and not person_class_ids:
        person_class_ids = {0}

    if ppe_model:
        print("helmet ids:", helmet_class_ids, "vest ids:", vest_class_ids)
        print("PPE filter ids:", ppe_class_ids)

    # ── [수정2] MBC 분류기 로드 (1명일 때 사용, TFLite/PyTorch 자동 선택) ──
    mbc_classifier = None
    if MBC_AVAILABLE and mbc_model_path:
        try:
            mbc_classifier = create_ppe_classifier(str(mbc_model_path), MBC_THRESHOLD)
            print(f"[통합] MBC 분류기 로드 완료: {mbc_model_path}")
        except Exception as e:
            print(f"[통합] MBC 분류기 로드 실패: {e}")
            mbc_classifier = None
    else:
        print("[통합] MBC 분류기 미사용 (모델 없음)")

    # ── 카메라 초기화 ──
    camera = LatestFrameCamera(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps_limit=30.0,
    )
    camera.start()
    print("[통합] 카메라 시작 (LatestFrameCamera)")
    time.sleep(1.0)

    # ── DangerDetector 초기화 ──
    danger_detector = DangerDetector(
        model_path=str(danger_sign_model_path) if danger_sign_model_path else "",
        sign_conf=SIGN_CONF,
        yolo_size=SIGN_YOLO_SIZE,
        zone_scale=DANGER_ZONE_SCALE,
        cache_seconds=SIGN_REFRESH_SECONDS,
        manual_zones=MANUAL_DANGER_ZONES,
    )
    print("[통합] DangerDetector 초기화 완료")

    # ── 스피커 초기화 ──
    danger_wav = SPEAKER_ASSETS_DIR / "DangerWarning.wav"
    ppe_wav = SPEAKER_ASSETS_DIR / "PPEWarning.wav"    
    speaker = DangerSpeaker(
        message="위험지역입니다",
        cooldown_seconds=SPEAKER_COOLDOWN,
        danger_audio_file=str(danger_wav) if danger_wav.exists() else None,
        ppe_audio_file=str(ppe_wav) if ppe_wav.exists() else None,
    )

    print(f"[통합] DangerSpeaker 초기화 완료 (wav: {danger_wav.exists()})")

    # ── [수정4] IR 센서 초기화 (기본 활성화) ──
    ir_sensor = None
    if IR_ENABLED and IR_AVAILABLE:
        try:
            ir_sensor = IRSensor(pin=IR_PIN, active_high=True)
            ir_sensor.start()
            print(f"[통합] IR 센서 시작 (GPIO {IR_PIN})")
        except Exception as e:
            print(f"[통합] IR 센서 시작 실패: {e} — IR 없이 상시 감지 모드로 동작")
            ir_sensor = None
    elif IR_ENABLED and not IR_AVAILABLE:
        print("[통합] IR 센서 활성화 설정이지만 RPi.GPIO 없음 — 상시 감지 모드로 동작")

    # ── 모든 모델이 없으면 카메라 전용 스트림 ──
    # person, PPE, danger sign 중 하나라도 있으면 메인 루프 진입
    if person_model is None and ppe_model is None and mbc_classifier is None and danger_sign_model_path is None:
        print("No models loaded, running camera-only stream.")
        min_frame_interval = 1.0 / max(MAX_STREAM_FPS, 1)
        fps = 0.0
        last_time = time.monotonic()
        while not stop_event.is_set():
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            frame = adjust_frame(frame)
            now = time.monotonic()
            elapsed = max(now - last_time, 0.001)
            last_time = now
            fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                data = {"people": 0, "ppe": 0, "signs": 0, "danger": False,
                        "fps": float(f"{fps:.1f}"), "persons": []}
                update_latest(jpeg.tobytes(), data)
            spent = time.monotonic() - now
            if spent < min_frame_interval:
                time.sleep(min_frame_interval - spent)
        _cleanup(camera, ir_sensor)
        return

    # ── 메인 감지 루프 ──
    cached_persons = []
    cached_ppe = {"helmets": [], "vests": []}  # [수정1] PPE 결과 캐시
    cached_mbc_results = {}  # [수정2] MBC 결과 캐시 {person_idx: (helmet, vest, h_prob, v_prob)}
    frame_index = 0
    fps = 0.0
    danger_hold = 0
    danger_zone_first_shown = 0.0
    danger_zone_visible = False
    last_time = time.monotonic()
    min_frame_interval = 1.0 / max(MAX_STREAM_FPS, 1)
    ir_triggered = False  # [수정4] IR 센서 트리거 상태

    print("[통합] 메인 감지 루프 시작")
    if ir_sensor:
        print("[통합] IR 센서 모드: 센서 감지 시에만 YOLO 실행")
    else:
        print("[통합] 상시 감지 모드 (IR 센서 없음)")

    while not stop_event.is_set():
        loop_start = time.monotonic()

        # ── [수정4] IR 센서: 감지 안 되면 카메라 프레임만 스트림 (추론 안 함) ──
        if ir_sensor is not None:
            currently_detected = ir_sensor.is_detected()

            if not currently_detected:
                # 사람 없음 → 추론 없이 카메라 프레임만 스트림
                if ir_triggered:
                    # 직전까지 감지 중이었으면 캐시 초기화
                    cached_persons = []
                    cached_ppe = {"helmets": [], "vests": []}
                    cached_mbc_results = {}
                    ir_triggered = False
                    print("[IR] 감지 해제 — 추론 중지")

                frame = camera.get_latest_frame()
                if frame is not None:
                    frame = adjust_frame(frame)
                    ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok:
                        now = time.monotonic()
                        elapsed = max(now - last_time, 0.001)
                        last_time = now
                        fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)
                        data = {"people": 0, "ppe": 0, "signs": 0, "danger": False,
                                "fps": float(f"{fps:.1f}"), "persons": []}
                        update_latest(jpeg.tobytes(), data)
                time.sleep(min_frame_interval)
                continue

            # 사람 감지됨 → 추론 실행
            if not ir_triggered:
                ir_triggered = True
                frame_index = 0  # 프레임 카운터 리셋 (첫 프레임에서 바로 감지하도록)
                print("[IR] 감지됨 — YOLO 추론 시작")

        # ── 프레임 가져오기 ──
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        frame = adjust_frame(frame)
        now = time.monotonic()
        img_h, img_w = frame.shape[:2]

        # ── Person 감지 (YOLO, 격프레임) ──
        should_update_persons = (
            person_model is not None
            and (frame_index % max(PERSON_DETECT_EVERY, 1) == 0 or not cached_persons)
        )

        if should_update_persons:
            person_result = person_model.predict(
                source=frame, imgsz=YOLO_SIZE, conf=PERSON_CONF, verbose=False
            )[0]
            cached_persons = []
            for box in person_result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in person_class_ids:
                    continue
                xyxy = tuple(float(v) for v in box.xyxy[0])
                cached_persons.append({"box": xyxy, "conf": float(box.conf[0])})
            cached_persons.sort(key=lambda item: item["box"][0])

        num_persons = len(cached_persons)

        # ── [수정2] 하이브리드 PPE 판정 ──
        # 1명: MBC 분류 (상체 크롭 → MobileNetV3)
        # 2명 이상: YOLO OD (best_p.pt)
        use_mbc = (num_persons == 1 and mbc_classifier is not None)
        use_od = (num_persons >= 2 and ppe_model is not None) or (num_persons == 1 and mbc_classifier is None and ppe_model is not None)

        # ── [수정1] PPE 감지도 격프레임 ──
        should_update_ppe = (frame_index % max(PPE_DETECT_EVERY, 1) == 0 or frame_index == 0)

        if should_update_ppe:
            if use_mbc:
                # MBC 방식: 1명의 상체 크롭 → 분류
                cached_mbc_results = {}
                person = cached_persons[0]
                x1, y1, x2, y2 = [int(v) for v in person["box"]]
                crop_box = compute_upper_body_box(x1, y1, x2, y2, img_w, img_h)
                if crop_box is not None and not _is_head_only(crop_box):
                    cx1, cy1, cx2, cy2 = crop_box
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        helmet, vest, h_prob, v_prob = mbc_classifier.classify(crop)
                        cached_mbc_results[0] = (helmet, vest, h_prob, v_prob)

                # MBC 실패 시 OD fallback에 쓸 수 있도록 PPE OD도 실행
                if 0 not in cached_mbc_results and ppe_model is not None:
                    ppe_result = ppe_model.predict(
                        source=frame, imgsz=YOLO_SIZE, conf=PPE_CONF, classes=ppe_class_ids, verbose=False
                    )[0]
                    helmets = []
                    vests = []
                    for box in ppe_result.boxes:
                        cls_id = int(box.cls[0])
                        xyxy = tuple(float(v) for v in box.xyxy[0])
                        item = {"box": xyxy, "conf": float(box.conf[0]), "center": center_of(xyxy)}
                        if cls_id in helmet_class_ids:
                            helmets.append(item)
                        elif cls_id in vest_class_ids:
                            vests.append(item)
                    cached_ppe = {"helmets": helmets, "vests": vests}
                else:
                    cached_ppe = {"helmets": [], "vests": []}

            elif use_od:
                # OD 방식: YOLO PPE 모델
                ppe_result = ppe_model.predict(
                    source=frame, imgsz=YOLO_SIZE, conf=PPE_CONF, classes=ppe_class_ids, verbose=False
                )[0]
                helmets = []
                vests = []
                for box in ppe_result.boxes:
                    cls_id = int(box.cls[0])
                    xyxy = tuple(float(v) for v in box.xyxy[0])
                    item = {"box": xyxy, "conf": float(box.conf[0]), "center": center_of(xyxy)}
                    if cls_id in helmet_class_ids:
                        helmets.append(item)
                    elif cls_id in vest_class_ids:
                        vests.append(item)
                cached_ppe = {"helmets": helmets, "vests": vests}
                # MBC 결과 비우기 (OD 사용 중)
                cached_mbc_results = {}

        # ── 위험 표지판 감지 ──
        person_boxes = [tuple(int(v) for v in p["box"]) for p in cached_persons]
        danger_result = danger_detector.detect(frame, person_boxes=person_boxes)
        danger_zones = [dz.bbox for dz in danger_result.danger_zones]

        # ── Person별 PPE + 위험 구역 판정 ──
        no_helmet_nums = []
        no_vest_nums = []
        danger_nums = []
        numbered_persons = []
        ppe_method = "mbc" if use_mbc else ("od" if use_od else "none")

        for idx, person in enumerate(cached_persons, start=1):
            person_box = person["box"]
            in_danger = any(boxes_overlap(person_box, zone) for zone in danger_zones)

            has_helmet = None
            has_vest = None
            person_method = ppe_method

            if use_mbc and (idx - 1) in cached_mbc_results:
                # MBC 결과 사용
                helmet, vest, _, _ = cached_mbc_results[idx - 1]
                has_helmet = helmet
                has_vest = vest
                person_method = "mbc"
            elif use_mbc and (idx - 1) not in cached_mbc_results and ppe_model is not None:
                # MBC 분류 실패 (크롭 너무 작음 등) → OD fallback
                match_box = expand_box(person_box, frame.shape, PERSON_BOX_PADDING)
                has_helmet = any(point_in_box(item["center"], match_box) for item in cached_ppe["helmets"])
                has_vest = any(point_in_box(item["center"], match_box) for item in cached_ppe["vests"])
                person_method = "od_fallback"
            elif use_od:
                # OD 결과 사용
                match_box = expand_box(person_box, frame.shape, PERSON_BOX_PADDING)
                has_helmet = any(point_in_box(item["center"], match_box) for item in cached_ppe["helmets"])
                has_vest = any(point_in_box(item["center"], match_box) for item in cached_ppe["vests"])
                person_method = "od"

            # PPE 모델 자체가 없는 경우 → unknown (경고 안 함)
            # PPE 모델은 있지만 분류 실패 → 미착용 처리 (안전 우선)
            ppe_unavailable = (has_helmet is None and has_vest is None and not use_mbc and not use_od)

            if ppe_unavailable:
                # 모델 미탑재: 판정 불가 상태
                has_helmet = None
                has_vest = None
                person_method = "unavailable"
            else:
                # 모델은 있지만 결과가 None (분류 실패) → 미착용 처리
                if has_helmet is None:
                    has_helmet = False
                if has_vest is None:
                    has_vest = False

            if not ppe_unavailable:
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
                "method": person_method,
            })

            # Person bbox 그리기
            x1, y1, x2, y2 = [int(v) for v in person_box]
            if ppe_unavailable:
                color = (128, 128, 128)  # 회색: PPE 판정 불가
            elif in_danger:
                color = (0, 0, 255)
            else:
                color = status_color(has_helmet, has_vest)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x1 + 22, y1 + 22), 20, color, -1)
            draw_text(frame, str(idx), (x1 + 11, y1 + 32), (255, 255, 255), 0.9, 2)
            if in_danger:
                draw_text(frame, "DANGER", (x1, max(25, y1 - 10)), (0, 0, 255), 0.75, 3)
            elif ppe_unavailable:
                draw_text(frame, "N/A", (x1, max(25, y1 - 10)), (128, 128, 128), 0.6, 2)

            # PPE 방식 표시
            if person_method == "unavailable":
                method_label = "PPE N/A"
            elif person_method == "mbc":
                method_label = "MBC"
            elif person_method == "od_fallback":
                method_label = "OD*"
            else:
                method_label = "OD"
            draw_text(frame, method_label, (x1, y2 + 18), (180, 180, 180), 0.45, 1)

        # ── 스피커 경고 (PPE 모델 미탑재 시에는 PPE 경고 안 함) ──
        for person in numbered_persons:
            if person["method"] == "unavailable":
                # PPE 모델 없음 → 위험 구역 경고만 가능
                if person["in_danger"]:
                    speaker.warn_danger_zone()
                continue
            if person["in_danger"] or not person["has_helmet"] or not person["has_vest"]:
                speaker.warn_person_status(
                    person_number=person["num"],
                    missing_helmet=not person["has_helmet"],
                    missing_vest=not person["has_vest"],
                    in_danger_zone=person["in_danger"],
                )

        # ── DANGER ZONE 표시 타이머 ──
        danger_hold = danger_hold + 1 if danger_nums else 0
        danger_active = danger_hold >= DANGER_HOLD_FRAMES

        if danger_active and not danger_zone_visible:
            danger_zone_first_shown = now
            danger_zone_visible = True
        elif not danger_active:
            danger_zone_visible = False

        show_danger_box = danger_zone_visible and (now - danger_zone_first_shown < DANGER_ZONE_DISPLAY_SECONDS)

        if show_danger_box:
            for zone in danger_zones:
                x1, y1, x2, y2 = [int(v) for v in zone]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                draw_text(frame, "DANGER ZONE", (x1, max(22, y1 - 8)), (0, 0, 255), 0.65, 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 54), (0, 0, 180), -1)
            draw_text(frame, "WARNING: PERSON NEAR DANGER SIGN", (18, 36), (255, 255, 255), 0.85, 2)

        # OD 모드일 때만 PPE bbox 그리기
        if use_od:
            for item in cached_ppe["helmets"]:
                x1, y1, x2, y2 = [int(v) for v in item["box"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            for item in cached_ppe["vests"]:
                x1, y1, x2, y2 = [int(v) for v in item["box"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 1)

        # ── FPS + JPEG + 상태 업데이트 ──
        elapsed = max(now - last_time, 0.001)
        last_time = now
        fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

        ppe_count = len(cached_ppe["helmets"]) + len(cached_ppe["vests"])
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            data = {
                "people": len(numbered_persons),
                "ppe": ppe_count,
                "signs": len(danger_result.signs),
                "danger": bool(danger_active),
                "fps": float(f"{fps:.1f}"),
                "ppe_method": ppe_method,
                "persons": [
                    {"num": p["num"], "has_helmet": p["has_helmet"],
                     "has_vest": p["has_vest"], "in_danger": p["in_danger"],
                     "method": p["method"]}
                    for p in numbered_persons
                ],
            }
            update_latest(jpeg.tobytes(), data)

        frame_index += 1
        spent = time.monotonic() - loop_start
        if spent < min_frame_interval:
            time.sleep(min_frame_interval - spent)

    _cleanup(camera, ir_sensor)


def _cleanup(camera, ir_sensor):
    try:
        camera.stop()
        print("[통합] 카메라 정지")
    except Exception:
        pass
    if ir_sensor is not None:
        try:
            ir_sensor.stop()
            print("[통합] IR 센서 정지")
        except Exception:
            pass


# =============================================================
# 웹 서버
# =============================================================

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
        if not index_path.exists():
            index_path = PROJECT_DIR / "ui" / "index.html"
        if index_path.exists():
            body = index_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        page = """<!doctype html>
<html><head><meta charset="utf-8"><title>Safe Eye</title>
<style>body{margin:0;background:#101010;color:white;font-family:Arial}
header{padding:12px 18px;background:#1e1e1e;font-size:20px;font-weight:700}
img{display:block;width:100vw;height:calc(100vh - 50px);object-fit:contain;background:#000}</style>
</head><body><header>Safe Eye Monitor</header>
<img src="/stream.mjpg" alt="camera stream"></body></html>"""
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_status(self):
        with latest_lock:
            data = latest_data.copy() if isinstance(latest_data, dict) else {"status": latest_status}
        if not data:
            data = {"status": latest_status}
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
    parser = argparse.ArgumentParser(description="Safe Eye 통합 웹 모니터 v2")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    worker = threading.Thread(target=detection_loop, daemon=True)
    worker.start()

    server = ThreadingHTTPServer((args.host, args.port), MonitorHandler)
    print(f"Open from PC: http://<raspberry-pi-ip>:{args.port}")
    print(f"Example: http://10.10.141.134:{args.port}")
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

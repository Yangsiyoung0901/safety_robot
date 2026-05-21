#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a lightweight detection test: show number of people and PPE status.

Usage:
  python tools/run_detection_test.py --camera 0
  python tools/run_detection_test.py --camera 0 --mbc

Behavior:
 - Tries to use `Main.detector.Detector` if available and person model exists.
 - Otherwise falls back to ultralytics YOLO person-only detection if available. 
 - If no model available, opens camera and shows frames with a "NO MODEL" overlay.
"""

from pathlib import Path
import time
import argparse
import sys

import cv2


def find_person_model():
    repo = Path(__file__).resolve().parents[1] / "models"
    candidates = [
        repo / "yolo11n.pt",
        repo / "yolov8n.pt",
        repo / "yolo8n.pt",
        repo / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def find_mbc_model():
    repo = Path(__file__).resolve().parents[1] / "models"
    candidates = [
        repo / "PPE_MobileNetV3Large.tflite",
        repo / "PPE_MobileNetV3Large_INT8.tflite",
        repo / "ppe_classifier.pth",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def try_detector_mode(person_model_path):
    try:
        from Main.detector import Detector
    except Exception:
        return None

    cfg = {
        "models": {
            "person_detector": person_model_path or "",
            "ppe_classifier": "",
        },
        "thresholds": {"person_confidence": 0.35, "ppe_threshold": 0.3},
        "crop": {"upper_ratio": 0.82, "expand": 0.08, "min_person_width": 40, "min_person_height": 80, "min_crop_height": 90, "min_crop_aspect": 0.9},
    }
    try:
        det = Detector(cfg)
        return det
    except Exception as e:
        print("Could not initialise Detector:", e)
    return None


def try_mbc_classifier(mbc_model_path, threshold, output_is_logits):
    if not mbc_model_path:
        return None
    try:
        repo = Path(__file__).resolve().parents[1]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from Main.detector import create_ppe_classifier
        return create_ppe_classifier(
            mbc_model_path,
            threshold=threshold,
            output_is_logits=output_is_logits,
        )
    except Exception as e:
        print("Could not initialise MBC classifier:", e)
        return None


def try_yolo_person(person_model_path):
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        if person_model_path:
            model = YOLO(person_model_path)
        else:
            # try remote / builtin small model name (may download)
            model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        print("Could not load YOLO model:", e)
        return None


def overlay_text(frame, lines):
    y = 20
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22


def compute_upper_body_box(person_box, frame_shape, expand=0.08, upper_ratio=0.82):
    x1, y1, x2, y2 = [int(v) for v in person_box]
    img_h, img_w = frame_shape[:2]
    person_w = x2 - x1
    person_h = y2 - y1
    upper_y2 = y1 + int(person_h * upper_ratio)

    pad_x = int(person_w * expand)
    pad_y_top = int(person_h * expand * 0.5)
    pad_y_bottom = int(person_h * expand * 0.25)

    cx1 = max(0, min(x1 - pad_x, img_w))
    cy1 = max(0, min(y1 - pad_y_top, img_h))
    cx2 = max(0, min(x2 + pad_x, img_w))
    cy2 = max(0, min(upper_y2 + pad_y_bottom, img_h))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--mbc", action="store_true", help="Run MobileNet/TFLite PPE test on the first detected person")
    parser.add_argument("--mbc-threshold", type=float, default=0.3)
    parser.add_argument("--mbc-output", choices=["logits", "prob"], default="logits")
    args = parser.parse_args()

    person_model = find_person_model()
    mbc_model = find_mbc_model()
    print("Person model:", person_model)
    print("MBC model:", mbc_model)

    mbc = try_mbc_classifier(
        mbc_model,
        threshold=args.mbc_threshold,
        output_is_logits=(args.mbc_output == "logits"),
    ) if args.mbc else None

    detector = None if args.mbc else try_detector_mode(person_model)
    if detector:
        mode = "detector"
        print("Using Detector from Main.detector")
    else:
        yolo = try_yolo_person(person_model)
        if yolo:
            mode = "yolo"
            print("Using ultralytics YOLO for person detection")
        else:
            mode = "camera_only"
            print("No person model available — running camera-only test")

    if args.mbc:
        if mbc is None:
            print("MBC test requested, but classifier could not be loaded.")
        else:
            print(f"Using MBC classifier ({args.mbc_output}, threshold={args.mbc_threshold})")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Could not open camera index", args.camera)
        sys.exit(1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            lines = []
            if mode == "detector":
                try:
                    dets = detector.detect(frame)
                    lines.append(f"People: {len(dets)}")
                    for idx, p in enumerate(dets, start=1):
                        h = p.helmet if p.helmet is not None else "?"
                        v = p.vest if p.vest is not None else "?"
                        lines.append(f"#{idx}: helmet={h} vest={v} method={getattr(p,'method','')}")
                except Exception as e:
                    lines.append("Detector error: " + str(e))
            elif mode == "yolo":
                try:
                    res = yolo.predict(source=frame, imgsz=320, conf=0.35, classes=[0], verbose=False)[0]
                    persons = []
                    for box in res.boxes:
                        persons.append(tuple(float(v) for v in box.xyxy[0]))
                    lines.append(f"People (YOLO): {len(persons)}")
                    if args.mbc and mbc is not None and persons:
                        persons.sort(key=lambda b: b[0])
                        person_box = persons[0]
                        crop_box = compute_upper_body_box(person_box, frame.shape)
                        x1, y1, x2, y2 = [int(v) for v in person_box]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                        if crop_box is None:
                            lines.append("MBC crop: None")
                        else:
                            cx1, cy1, cx2, cy2 = crop_box
                            crop = frame[cy1:cy2, cx1:cx2]
                            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 180, 0), 2)
                            helmet, vest, h_prob, v_prob = mbc.classify(crop)
                            lines.append(f"MBC helmet={helmet} p={h_prob:.4f}")
                            lines.append(f"MBC vest={vest} p={v_prob:.4f}")
                            print(
                                f"[MBC TEST] crop={crop.shape} helmet={helmet} "
                                f"h_prob={h_prob:.4f} vest={vest} v_prob={v_prob:.4f}"
                            )
                except Exception as e:
                    lines.append("YOLO error: " + str(e))
            else:
                lines.append("NO MODEL — camera only")

            # overlay and show
            overlay_text(frame, lines[:6])
            if not args.no_window:
                cv2.imshow("detection_test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                # just print summary periodically
                print(" | ".join(lines))
                time.sleep(0.5)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

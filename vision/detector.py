"""
detector.py — Person / PPE / Danger 통합 파이프라인

역할:
  1. YOLO Person Detection — 모든 상태에서 상시 실행
  2. PPE 판정 (하이브리드)
     - 1명: 상체 크롭 → MobileNetV3 Classification
     - 2명 이상: PPE OD 모델로 helmet·vest bbox 검출
  3. Danger Detection — 위험 표지 감지 + 근접 판정
  
다른 모듈에서 사용:
  from vision.detector import Detector, PersonDetection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


# =============================================================
# 데이터 클래스
# =============================================================

@dataclass
class PersonDetection:
    """한 사람의 감지 + PPE + 위험구역 판정 결과"""
    bbox: tuple[int, int, int, int]          # person bbox (x1, y1, x2, y2)
    confidence: float
    crop_box: tuple[int, int, int, int] | None = None
    helmet: bool | None = None               # True=착용, False=미착용, None=미판정
    vest: bool | None = None
    helmet_prob: float = 0.0
    vest_prob: float = 0.0
    in_danger: bool = False                  # 위험 구역 근접 여부
    method: str = ""                         # "classification" / "od" / ""


# =============================================================
# 상체 크롭 함수 (Classification 방식에서 사용)
# =============================================================

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def compute_upper_body_box(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    expand: float = 0.08,
    upper_ratio: float = 0.82,
) -> tuple[int, int, int, int] | None:
    """
    person bbox에서 상체 영역을 계산한다.
    상체 = bbox 상단부터 upper_ratio(82%) 높이 + 패딩.
    """
    person_w = x2 - x1
    person_h = y2 - y1
    upper_y2 = y1 + int(person_h * upper_ratio)

    pad_x = int(person_w * expand)
    pad_y_top = int(person_h * expand * 0.5)
    pad_y_bottom = int(person_h * expand * 0.25)

    cx1 = _clamp(x1 - pad_x, 0, img_w)
    cy1 = _clamp(y1 - pad_y_top, 0, img_h)
    cx2 = _clamp(x2 + pad_x, 0, img_w)
    cy2 = _clamp(upper_y2 + pad_y_bottom, 0, img_h)

    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def _is_head_only(
    crop_box: tuple[int, int, int, int],
    min_crop_height: int = 90,
    min_crop_aspect: float = 0.90,
) -> bool:
    """크롭이 머리만 잡힌 것인지 판단 (너무 작거나 넓적하면 제외)"""
    x1, y1, x2, y2 = crop_box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return True
    if h < min_crop_height:
        return True
    if (h / w) < min_crop_aspect:
        return True
    return False


# =============================================================
# PPE Classification 모델 (1명일 때 사용)
# =============================================================

class PPEClassifier:
    """
    MobileNetV3Large 기반 PPE 착용 여부 분류.
    입력: 크롭된 상체 이미지 (BGR)
    출력: (helmet_worn, vest_worn, helmet_prob, vest_prob)
    """

    def __init__(self, model_path: str, device: torch.device, threshold: float = 0.3):
        self.device = device
        self.threshold = threshold
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self, model_path: str) -> nn.Module:
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def classify(self, crop_bgr: np.ndarray) -> tuple[bool, bool, float, float]:
        """크롭 이미지 → (helmet_worn, vest_worn, helmet_prob, vest_prob)"""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu()

        h_prob = probs[0].item()
        v_prob = probs[1].item()
        return h_prob >= self.threshold, v_prob >= self.threshold, h_prob, v_prob


# =============================================================
# 통합 파이프라인
# =============================================================

class Detector:
    """
    Person / PPE / Danger 통합 감지 파이프라인.

    하이브리드 PPE 판정:
      - 1명 감지: 상체 크롭 → MobileNetV3 Classification
      - 2명 이상: PPE OD 모델에서 별도 처리 (팀원 코드 연동)
    """

    def __init__(self, cfg: dict):
        model_cfg = cfg["models"]
        thresh_cfg = cfg["thresholds"]
        crop_cfg = cfg["crop"]

        self.person_conf = thresh_cfg["person_confidence"]

        # 크롭 파라미터
        self.upper_ratio = crop_cfg["upper_ratio"]
        self.expand = crop_cfg["expand"]
        self.min_person_w = crop_cfg["min_person_width"]
        self.min_person_h = crop_cfg["min_person_height"]
        self.min_crop_h = crop_cfg["min_crop_height"]
        self.min_crop_aspect = crop_cfg["min_crop_aspect"]

        # --- Person Detector (YOLO) ---
        print(f"[Detector] Person 모델 로드: {model_cfg['person_detector']}")
        self.person_model = YOLO(model_cfg["person_detector"])

        # --- PPE Classifier (MobileNetV3, 1명일 때 사용) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ppe_cls_path = model_cfg.get("ppe_classifier", "")
        if ppe_cls_path and Path(ppe_cls_path).exists():
            print(f"[Detector] PPE Classification 모델 로드: {ppe_cls_path}")
            self.ppe_classifier = PPEClassifier(
                ppe_cls_path, self.device, thresh_cfg["ppe_threshold"]
            )
        else:
            print("[Detector] PPE Classification 모델 없음")
            self.ppe_classifier = None

        # --- PPE OD 모델 (2명 이상일 때 사용, 팀원 코드에서 처리) ---
        # ppe_od_model은 팀원 코드(safe_eye_monitor.py)에서 로드·실행
        # 여기서는 경로만 보관
        self.ppe_od_path = model_cfg.get("ppe_od_model", "")

        # --- Danger Detector (팀원 코드에서 처리) ---
        # danger_detector는 팀원 코드에서 로드·실행
        self.danger_path = model_cfg.get("danger_detector", "")

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """
        한 프레임에서 Person 감지 + 하이브리드 PPE 판정을 수행한다.

        하이브리드 분기:
          - 1명: 상체 크롭 → MobileNetV3 Classification → helmet/vest 채움
          - 2명 이상: helmet/vest를 None으로 남김 → OD에서 별도 처리

        Args:
            frame: BGR 이미지 (numpy array)

        Returns:
            PersonDetection 리스트 (왼쪽→오른쪽 정렬)
        """
        img_h, img_w = frame.shape[:2]

        # ── 1) YOLO Person 검출 ──
        results = self.person_model.predict(
            source=frame,
            imgsz=320,
            conf=self.person_conf,
            iou=0.5,
            classes=[0],
            verbose=False,
        )[0]

        # 유효한 person bbox 수집
        persons = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            conf = float(box.conf[0].cpu().item())

            if (x2 - x1) < self.min_person_w or (y2 - y1) < self.min_person_h:
                continue
            persons.append((x1, y1, x2, y2, conf))

        # 왼쪽→오른쪽 정렬 (x1 기준)
        persons.sort(key=lambda p: p[0])

        # ── 2) 하이브리드 PPE 분기 ──
        use_classification = (len(persons) == 1)
        detections: list[PersonDetection] = []

        for x1, y1, x2, y2, conf in persons:
            det = PersonDetection(bbox=(x1, y1, x2, y2), confidence=conf)

            if use_classification and self.ppe_classifier is not None:
                # --- 1명: 크롭 → MobileNetV3 분류 ---
                crop_box = compute_upper_body_box(
                    x1, y1, x2, y2, img_w, img_h,
                    self.expand, self.upper_ratio,
                )
                if crop_box is not None and not _is_head_only(
                    crop_box, self.min_crop_h, self.min_crop_aspect
                ):
                    det.crop_box = crop_box
                    cx1, cy1, cx2, cy2 = crop_box
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        helmet, vest, h_prob, v_prob = self.ppe_classifier.classify(crop)
                        det.helmet = helmet
                        det.vest = vest
                        det.helmet_prob = h_prob
                        det.vest_prob = v_prob
                        det.method = "classification"

            # 2명 이상이면 helmet/vest = None, method = ""
            # → 팀원 코드의 OD 방식이 처리
            detections.append(det)

        return detections

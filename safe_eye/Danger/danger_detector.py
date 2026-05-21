# -*- coding: utf-8 -*-
"""
danger_detector.py — 위험 표지판 인식 + 위험 구역 근접 판정 모듈

        역할:
    1. best_p.pt 로 위험 표지판 bbox 검출
  2. 표지판 bbox를 기준으로 DANGER ZONE 영역 확장
  3. person bbox와 DANGER ZONE 겹침 여부 판정

다른 모듈에서 사용:
  from Danger import DangerDetector
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ultralytics 오프라인 모드 (네트워크 없는 환경 대비)
os.environ.setdefault("YOLO_OFFLINE", "1")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# =============================================================
# 데이터 클래스
# =============================================================

@dataclass
class SignDetection:
    """표지판 감지 결과"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str = "sign"


@dataclass
class DangerZone:
    """표지판으로부터 확장된 위험 구역"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    source_sign: SignDetection


@dataclass
class DangerResult:
    """위험 판정 결과 (프레임 단위)"""
    signs: List[SignDetection]              # 감지된 표지판 목록
    danger_zones: List[DangerZone]          # 확장된 위험 구역 목록
    persons_in_danger: List[int]            # 위험 구역에 진입한 person 인덱스 목록
    danger_active: bool = False             # 위험 구역 진입자 존재 여부


# =============================================================
# 유틸 함수
# =============================================================

def _boxes_overlap(a: Tuple, b: Tuple) -> bool:
    """두 bbox가 겹치는지 판정"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


def _expand_zone(
    sign_box: Tuple[float, float, float, float],
    frame_h: int,
    frame_w: int,
    scale: float = 3.0,
) -> Tuple[float, float, float, float]:
    """
    표지판 bbox를 기준으로 DANGER ZONE 영역을 확장한다.
    scale=3.0이면 표지판 크기의 3배로 확장.
    """
    x1, y1, x2, y2 = sign_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = max(1.0, x2 - x1) * scale
    bh = max(1.0, y2 - y1) * scale
    return (
        max(0, cx - bw / 2),
        max(0, cy - bh / 2),
        min(frame_w - 1, cx + bw / 2),
        min(frame_h - 1, cy + bh / 2),
    )


# =============================================================
# 메인 클래스
# =============================================================

class DangerDetector:
    """
    위험 표지판 인식 + 위험 구역 근접 판정.

    사용법:
        detector = DangerDetector(model_path="models/best_p.pt")
        result = detector.detect(frame, person_boxes=[(x1,y1,x2,y2), ...])
        if result.danger_active:
            print("위험 구역 진입:", result.persons_in_danger)

    설정값:
        model_path: best_p.pt 경로
        sign_conf: 표지판 감지 신뢰도 임계값 (기본 0.45)
        yolo_size: YOLO 입력 크기 (기본 416)
        zone_scale: 표지판 대비 위험 구역 확장 배율 (기본 3.0)
        cache_seconds: 표지판 위치 캐시 유지 시간 (기본 10초, 고정 카메라용)
        manual_zones: 수동 지정 위험 구역 리스트 [(x1,y1,x2,y2), ...]
    """

    def __init__(
        self,
        model_path: str = "models/best_p.pt",
        sign_conf: float = 0.45,
        yolo_size: int = 416,
        zone_scale: float = 3.0,
        cache_seconds: float = 10.0,
        manual_zones: Optional[List[Tuple]] = None,
    ):
        self.sign_conf = sign_conf
        self.yolo_size = yolo_size
        self.zone_scale = zone_scale
        self.cache_seconds = cache_seconds
        self.manual_zones = manual_zones or []

        # 모델 로드
        self.model = None
        self.sign_class_ids: set = set()
        self._load_model(model_path)

        # 캐시 (고정 카메라에서 표지판은 움직이지 않으므로)
        self._cached_signs: List[SignDetection] = []
        self._last_detect_time: float = 0.0

    def _load_model(self, model_path: str) -> None:
        """best_p.pt 모델을 로드하고 클래스 ID를 매핑한다."""
        path = Path(model_path)
        if not path.exists():
            print(f"[DangerDetector] 모델 파일 없음: {model_path}")
            return
        if YOLO is None:
            print("[DangerDetector] ultralytics 미설치 — 표지판 감지 비활성화")
            return

        print(f"[DangerDetector] 모델 로드: {model_path}")
        self.model = YOLO(str(path))

        # 클래스 이름 매핑 — best_p.pt의 클래스 이름에 맞춰 자동 탐색
        target_names = {"danger_sign", "warning_sign", "sign", "dangersign", "warningsign"}
        for cls_id, name in self.model.names.items():
            normalized = name.lower().replace("-", "").replace("_", "").replace(" ", "")
            for target in target_names:
                if normalized == target.replace("_", ""):
                    self.sign_class_ids.add(cls_id)

        # 클래스를 못 찾으면 전체 클래스 사용 (단일 클래스 모델인 경우)
        if not self.sign_class_ids:
            self.sign_class_ids = set(self.model.names.keys())
            print(f"[DangerDetector] 클래스 자동 매핑 실패 → 전체 사용: {self.model.names}")
        else:
            matched = {self.model.names[i] for i in self.sign_class_ids}
            print(f"[DangerDetector] 매핑된 클래스: {matched}")

    def _detect_signs(self, frame: np.ndarray) -> List[SignDetection]:
        """프레임에서 표지판을 감지한다."""
        if self.model is None:
            return []

        result = self.model.predict(
            source=frame,
            imgsz=self.yolo_size,
            conf=self.sign_conf,
            verbose=False,
        )[0]

        signs = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.sign_class_ids:
                continue
            xyxy = tuple(float(v) for v in box.xyxy[0])
            conf = float(box.conf[0])
            name = self.model.names.get(cls_id, "sign")
            signs.append(SignDetection(bbox=xyxy, confidence=conf, class_name=name))

        return signs

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        force_refresh: bool = False,
    ) -> DangerResult:
        """
        프레임에서 표지판 감지 + 위험 구역 판정을 수행한다.

        Args:
            frame: BGR 이미지 (numpy array)
            person_boxes: person bbox 리스트 [(x1,y1,x2,y2), ...]
                          None이면 위험 구역만 계산하고 진입 판정은 건너뜀
            force_refresh: True면 캐시 무시하고 즉시 재감지

        Returns:
            DangerResult
        """
        now = time.monotonic()
        h, w = frame.shape[:2]
        person_boxes = person_boxes or []

        # --- 표지판 감지 (캐시 사용) ---
        should_refresh = (
            force_refresh
            or not self._cached_signs
            or now - self._last_detect_time >= self.cache_seconds
        )
        if should_refresh:
            new_signs = self._detect_signs(frame)
            # 재탐지 결과가 비어도 캐시를 갱신한다.
            # 기존: new_signs가 빈 리스트면 이전 캐시 유지 → 오탐 1회가 영구 지속
            # 수정: 빈 결과도 반영하여 표지판이 사라지면 위험 구역도 해제
            self._cached_signs = new_signs
            self._last_detect_time = now

        # --- 위험 구역 생성 ---
        danger_zones: List[DangerZone] = []

        # 수동 지정 구역
        for zone in self.manual_zones:
            dummy_sign = SignDetection(bbox=zone, confidence=1.0, class_name="manual")
            danger_zones.append(DangerZone(bbox=zone, source_sign=dummy_sign))

        # 표지판 기반 확장 구역
        for sign in self._cached_signs:
            expanded = _expand_zone(sign.bbox, h, w, self.zone_scale)
            danger_zones.append(DangerZone(bbox=expanded, source_sign=sign))

        # --- person bbox와 위험 구역 겹침 판정 ---
        persons_in_danger: List[int] = []
        for i, pbox in enumerate(person_boxes):
            for dz in danger_zones:
                if _boxes_overlap(pbox, dz.bbox):
                    persons_in_danger.append(i)
                    break

        return DangerResult(
            signs=list(self._cached_signs),
            danger_zones=danger_zones,
            persons_in_danger=persons_in_danger,
            danger_active=len(persons_in_danger) > 0,
        )

    def get_cached_signs(self) -> List[SignDetection]:
        """현재 캐시된 표지판 목록 반환"""
        return list(self._cached_signs)

    def clear_cache(self) -> None:
        """표지판 캐시 초기화 (카메라 위치 변경 시)"""
        self._cached_signs = []
        self._last_detect_time = 0.0
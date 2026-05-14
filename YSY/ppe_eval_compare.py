"""
================================================================================
PPE 성능 비교 평가 스크립트  (임시 / 중간 발표용)
================================================================================
OD 파이프라인 vs MBC 파이프라인의 PPE(헬멧·조끼) 판별 성능 비교.

[흐름]
  1. yolo11m 으로 Person Detection  — 한 번만 실행, GT/OD/MBC 가 공유
  2. OD 라벨 + person 박스  -> Ground Truth 자동 생성
  3. OD 파이프라인 : best.pt 로 head/helmet/vest 검출 -> person 박스에 귀속
  4. MBC 파이프라인: person 박스 상체 크롭 -> float TFLite 분류
  5. GT 대비 정확도 / FNR 평가 + 시각화(콘솔표 + PNG 그래프)

[전제 / 결정사항]  (대화로 확정한 내용)
  - person 박스는 yolo11m 한 번 실행분을 GT/OD/MBC 가 공유 -> "인원 수 정확도"는 평가 제외
    · detector/크롭/conf 모두 팀원 C 의 Crop_with_labels.ipynb(= MBC 학습 크롭) 와 일치시킴
    · person conf 0.25, IoU 0.5 중복 박스 제거 — 학습 크롭 파이프라인과 동일
  - GT 헬멧 : helmet 라벨(0) -> 1,  no-helmet 라벨(2) -> 0,  둘 다 없으면 헬멧 평가에서 제외
  - GT 조끼 : vest 라벨(1) -> 1,  없으면 0 (부재 추론. no-vest 라벨(3)은 대부분 생략 -> 미사용)
  - OD 헬멧 : helmet bbox 귀속 -> 1, 부재 -> 0 (부재 추론).
              best.pt 가 head 클래스를 실제로 검출 못 해 head 기반 판정은 불가
  - OD 조끼 : vest bbox 귀속 -> 1, 부재 -> 0 (부재 추론)
  - 귀속 기준 : PPE bbox 면적의 50% 이상이 person bbox(패딩 10%) 안에 포함
               (참고: 팀원 C 학습 라벨은 크롭 영역 기준 매칭 — GT 는 person 박스 기준, 미세 차이 존재)
  - MBC 크롭 : 팀원 C 의 crop_upper_body 와 동일 — 종횡비 분기(전신 상위 65% / 중간 +10% / 상체 +5%),
               좌우·상단 패딩 없음. 이후 resize_with_padding 224x224 (학습과 동일)
  - 임계값 전부 0.5  (MBC FNR 이 높으면 MBC_THRESHOLD 0.6 도 시험 권장)

[의존성]
  pip install ultralytics tensorflow opencv-python numpy matplotlib scikit-learn
================================================================================
"""

import os
import re
import csv
import glob
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 창 없이 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 한글 폰트 설정 — 그래프 라벨 깨짐 / Glyph missing 경고 방지
# Windows: Malgun Gothic, macOS: AppleGothic, Linux: NanumGothic 중 있는 것 사용
_installed = {f.name for f in fm.fontManager.ttflist}
for _kfont in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"):
    if _kfont in _installed:
        plt.rcParams["font.family"] = _kfont
        print(f"[font] 한글 폰트 사용: {_kfont}")
        break
else:
    print("[font] 한글 폰트를 찾지 못함 — 그래프 한글이 깨질 수 있음")
plt.rcParams["axes.unicode_minus"] = False   # 음수 기호 깨짐 방지

from ultralytics import YOLO
try:
    import tensorflow as tf
except ImportError:
    raise SystemExit("tensorflow 가 필요합니다:  pip install tensorflow")


# ==============================================================================
# ★ 설정 — 본인 환경에 맞게 수정
# ==============================================================================
PERSON_MODEL_PATH = "yolo11m.pt"                      # MBC 학습 크롭과 동일 detector
OD_MODEL_PATH     = r"D:\AIProject\YSY\best.pt"
MBC_MODEL_PATH    = r"D:\AIProject\YSY\PPE_MobileNetV3Large.tflite"   # float (INT8 은 정확도 손실 큼)

# 테스트셋 — (표시이름, images 폴더, labels 폴더)
DATASETS = [
    ("small (2인 이하)",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\small\images",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\small\labels"),
    ("large (3인 이상)",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\large\images",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\large\labels"),
]

OUTPUT_DIR = r"D:\AIProject\YSY\eval_results"         # 그래프 / CSV 저장 위치

# OD 라벨 클래스 ID
LABEL_HELMET    = 0
LABEL_VEST      = 1
LABEL_NO_HELMET = 2
# class 3 (no-vest) 은 대부분 생략 -> 신뢰 불가 -> 사용하지 않음

# 임계값 / 파라미터  — 팀원 C 의 Crop_with_labels.ipynb 와 동일하게 맞춤
PERSON_CONF        = 0.25     # 팀원 C: PERSON_CONFIDENCE = 0.25
PERSON_IMGSZ       = 640      # 팀원 C: imgsz 미지정 -> ultralytics 기본값 640
OD_CONF            = 0.5
OD_IMGSZ           = 640
MBC_THRESHOLD      = 0.5
CONTAINMENT_RATIO  = 0.5      # PPE bbox 면적이 person bbox 안에 포함되는 최소 비율
PERSON_BOX_PADDING = 0.10     # 귀속 판정용 person bbox 패딩 (GT/OD attribution 용)
MBC_INPUT_SIZE     = 224
IOU_DUPLICATE_THRESHOLD = 0.5 # 팀원 C: 사람 bbox 중복 제거 IoU 임계값

# MBC 상체 크롭 파라미터 — 팀원 C 의 crop_upper_body 와 동일
# 전신(종횡비>=2.0)은 상위 65%, 그 외는 아래로 확장. 좌우/상단 패딩 없음.
UPPER_BODY_RATIO  = 0.65

# KPI 기준선 (그래프 표시용, 기획서 §11)
KPI_HELMET_ACC = 0.80
KPI_VEST_ACC   = 0.70
KPI_FNR        = 0.15

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


# ==============================================================================
# 기하 / 입출력 헬퍼
# ==============================================================================
def containment_ratio(inner, outer):
    """inner 박스 면적 중 outer 박스 안에 포함된 비율 (0.0 ~ 1.0)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    iw = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    ih = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inter = iw * ih
    inner_area = max((ix2 - ix1) * (iy2 - iy1), 1e-6)
    return inter / inner_area


def expand_box(box, img_w, img_h, ratio):
    """person 박스에 상하좌우 padding 적용 (이미지 경계 안으로 클램프)."""
    x1, y1, x2, y2 = box
    px = (x2 - x1) * ratio
    py = (y2 - y1) * ratio
    return (max(0.0, x1 - px), max(0.0, y1 - py),
            min(img_w - 1.0, x2 + px), min(img_h - 1.0, y2 + py))


def parse_label_file(path, img_w, img_h):
    """YOLO OD 라벨(.txt) -> [(class_id, (x1,y1,x2,y2)), ...]  절대좌표."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append((cls, (x1, y1, x2, y2)))
    return boxes


def calculate_iou(box_a, box_b):
    """두 bbox 의 IoU (x1,y1,x2,y2). 팀원 C 코드와 동일."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def remove_duplicate_boxes(persons, iou_threshold=IOU_DUPLICATE_THRESHOLD):
    """IoU 가 높은 중복 person bbox 제거. 팀원 C 코드와 동일.
       persons: [{'box': (x1,y1,x2,y2), 'conf': float}, ...]"""
    if len(persons) <= 1:
        return persons
    keep, used = [], set()
    ordered = sorted(persons, key=lambda p: p["conf"], reverse=True)
    for i, p in enumerate(ordered):
        if i in used:
            continue
        keep.append(p)
        for j in range(i + 1, len(ordered)):
            if j in used:
                continue
            if calculate_iou(p["box"], ordered[j]["box"]) > iou_threshold:
                used.add(j)
    return keep


def crop_upper_body_box(person_box, img_w, img_h):
    """person bbox -> 상체 크롭 영역 (x1,y1,x2,y2).
       팀원 C 의 crop_upper_body 와 동일: 종횡비별 분기, 좌우/상단 패딩 없음."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    w = x2 - x1
    aspect = (h / w) if w > 0 else 1.0
    if aspect >= 2.0:                        # 전신 -> 상위 65%
        crop_y2 = y1 + h * UPPER_BODY_RATIO
    elif aspect >= 1.2:                      # 중간 -> 아래로 10% 확장
        crop_y2 = min(y2 + h * 0.10, img_h)
    else:                                    # 이미 상체 -> 아래로 5% 확장
        crop_y2 = min(y2 + h * 0.05, img_h)
    cx1 = max(0.0, x1)
    cy1 = max(0.0, y1)
    cx2 = min(float(img_w), x2)
    cy2 = min(float(img_h), crop_y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return (cx1, cy1, cx2, cy2)


def resize_with_padding(img, size):
    """긴 변 기준 스케일 후 검정 캔버스 중앙 배치 (PPE_Classification 노트북 방식)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


# ==============================================================================
# 모델 래퍼
# ==============================================================================
def build_od_class_map(od_model):
    """OD 모델의 class id -> 역할('helmet'/'head'/'vest') 매핑. head = no-helmet 취급."""
    role = {}
    for cid, name in od_model.names.items():
        n = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if n in ("helmet", "hardhat", "hard_hat", "safety_helmet"):
            role[cid] = "helmet"
        elif n in ("head", "no_helmet", "nohelmet"):
            role[cid] = "head"          # head = no-helmet 과 동일
        elif n in ("vest", "safety_vest", "safetyvest"):
            role[cid] = "vest"
    return role


class MBCModel:
    """INT8 TFLite PPE 분류기 래퍼. 입력 양자화 / 출력 역양자화를 모델에서 읽어 처리."""

    def __init__(self, model_path):
        self.interp = self._load_interpreter(model_path)
        self.in_det = self.interp.get_input_details()
        self.out_det = self.interp.get_output_details()
        self.in_dtype = self.in_det[0]["dtype"]
        self.out_dtype = self.out_det[0]["dtype"]
        self.in_scale, self.in_zp = self.in_det[0]["quantization"]
        self.out_scale, self.out_zp = self.out_det[0]["quantization"]
        print(f"  MBC 입력 dtype={self.in_dtype.__name__}, "
              f"양자화(scale={self.in_scale:.5f}, zp={self.in_zp})")

    def predict(self, crop_bgr):
        """크롭 이미지(BGR) -> (helmet_prob, vest_prob)."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inp = resize_with_padding(rgb, MBC_INPUT_SIZE).astype(np.float32)

        # 입력 양자화 : q = real/scale + zero_point
        if self.in_dtype in (np.int8, np.uint8) and self.in_scale not in (0, None):
            q = np.round(inp / self.in_scale + self.in_zp)
            info = np.iinfo(self.in_dtype)
            q = np.clip(q, info.min, info.max).astype(self.in_dtype)
            tensor = q[np.newaxis, ...]
        else:
            tensor = inp[np.newaxis, ...]

        self.interp.set_tensor(self.in_det[0]["index"], tensor)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det[0]["index"])[0].astype(np.float32)

        # 출력 역양자화 : real = scale * (q - zero_point)
        if self.out_dtype in (np.int8, np.uint8) and self.out_scale not in (0, None):
            out = self.out_scale * (out - self.out_zp)

        return float(out[0]), float(out[1])

    @staticmethod
    def _load_interpreter(model_path):
        """INT8 모델 로드. XNNPACK delegate 가 일부 노드(MobileNetV3 연산)에서
           prepare 실패하는 경우가 있어, 자동 delegate 적용을 끄고 로드한다."""
        OpResolver = tf.lite.experimental.OpResolverType
        last_err = None
        for resolver in (OpResolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                         OpResolver.BUILTIN_REF):
            try:
                interp = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_op_resolver_type=resolver)
                interp.allocate_tensors()
                print(f"  MBC interpreter 로드 OK (resolver={resolver.name})")
                return interp
            except (RuntimeError, ValueError) as e:
                last_err = e
                print(f"  resolver={resolver.name} 실패 -> 다음 방식 시도")
        raise SystemExit(f"MBC 모델 로드 실패: {last_err}")


# ==============================================================================
# 파이프라인 — person 박스 1개에 대한 판정
# ==============================================================================
def make_gt(person_box, label_boxes, img_w, img_h):
    """OD 라벨로 GT 생성. 반환: (gt_helmet ∈ {0,1,None}, gt_vest ∈ {0,1})."""
    padded = expand_box(person_box, img_w, img_h, PERSON_BOX_PADDING)
    helmet = [b for c, b in label_boxes if c == LABEL_HELMET]
    nohelm = [b for c, b in label_boxes if c == LABEL_NO_HELMET]
    vest = [b for c, b in label_boxes if c == LABEL_VEST]

    has_helmet = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in helmet)
    has_nohelm = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in nohelm)
    has_vest = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in vest)

    if has_helmet:                    # helmet 우선
        gt_helmet = 1
    elif has_nohelm:
        gt_helmet = 0
    else:
        gt_helmet = None              # 헬멧 평가에서 제외
    gt_vest = 1 if has_vest else 0
    return gt_helmet, gt_vest


def od_predict(person_box, od_boxes, img_w, img_h):
    """OD 모델 출력으로 판정. od_boxes: [(role, box), ...]. 반환: (helmet ∈ {0,1}, vest ∈ {0,1}).

    주의: 학습된 best.pt 가 head(no-helmet) 클래스를 실질적으로 검출하지 못해(conf 0.05
    에서도 미출력), 헬멧 미착용을 'helmet bbox 부재'로 추론한다. vest 와 동일 방식이며,
    팀원 원본 코드(safe_eye.py)도 실질적으로 이 방식이었다.
    한계: best.pt 가 헬멧을 놓친 경우와 실제 미착용을 구분하지 못함 (발표 시 명시 필요)."""
    padded = expand_box(person_box, img_w, img_h, PERSON_BOX_PADDING)
    helmet = [b for r, b in od_boxes if r == "helmet"]
    vest = [b for r, b in od_boxes if r == "vest"]

    has_helmet = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in helmet)
    has_vest = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in vest)

    od_helmet = 1 if has_helmet else 0      # 부재 추론 (head 클래스 사용 불가)
    od_vest = 1 if has_vest else 0
    return od_helmet, od_vest


def mbc_predict(frame, person_box, mbc_model):
    """상체 크롭 -> MBC 분류. 반환: (helmet ∈ {0,1}, vest ∈ {0,1}).
    크롭은 팀원 C 의 crop_upper_body 와 동일 (종횡비 분기, 패딩 없음)."""
    img_h, img_w = frame.shape[:2]
    crop_box = crop_upper_body_box(person_box, img_w, img_h)
    if crop_box is None:
        crop_box = person_box                     # 퇴화 시 person 박스 전체로 폴백
    x1, y1, x2, y2 = (int(round(v)) for v in crop_box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0, 0
    h_prob, v_prob = mbc_model.predict(crop)
    return int(h_prob >= MBC_THRESHOLD), int(v_prob >= MBC_THRESHOLD)


# ==============================================================================
# 데이터셋 처리
# ==============================================================================
def list_images(images_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(files)


def process_dataset(name, images_dir, labels_dir, person_model, od_model,
                    od_role_map, mbc_model):
    """한 데이터셋 폴더 전체 처리 -> per-person 레코드 리스트 + 타이밍/진단 정보."""
    records = []
    timing = {"person": 0.0, "od": 0.0, "mbc": 0.0, "images": 0, "persons": 0}
    diag = {"no_person": 0, "no_label": 0, "empty_person": 0}

    image_files = list_images(images_dir)
    print(f"\n[{name}]  이미지 {len(image_files)}장")
    if not image_files:
        print(f"  ⚠️  이미지를 찾을 수 없음: {images_dir}")
        return records, timing, diag

    for i, img_path in enumerate(image_files, start=1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  ⚠️  읽기 실패: {os.path.basename(img_path)}")
            continue
        img_h, img_w = frame.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        label_path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.exists(label_path):
            diag["no_label"] += 1
        label_boxes = parse_label_file(label_path, img_w, img_h)

        # 1) Person Detection (한 번 — GT/OD/MBC 공유)
        #    팀원 C 의 Crop_with_labels.ipynb 와 동일: conf=0.25 + IoU 중복 제거
        t0 = time.perf_counter()
        pres = person_model.predict(source=frame, imgsz=PERSON_IMGSZ,
                                    conf=PERSON_CONF, classes=[0], verbose=False)[0]
        timing["person"] += time.perf_counter() - t0
        persons = [{"box": tuple(float(v) for v in b.xyxy[0]),
                    "conf": float(b.conf[0])} for b in pres.boxes]
        persons = remove_duplicate_boxes(persons)
        person_boxes = [p["box"] for p in persons]
        person_boxes.sort(key=lambda b: b[0])      # 좌->우 순으로 ID 부여
        if not person_boxes:
            diag["no_person"] += 1
            timing["images"] += 1
            continue

        timing["persons"] += len(person_boxes)

        # 2) OD 모델 추론 (이미지 1회)
        t0 = time.perf_counter()
        ores = od_model.predict(source=frame, imgsz=OD_IMGSZ,
                                conf=OD_CONF, verbose=False)[0]
        timing["od"] += time.perf_counter() - t0
        od_boxes = []
        for b in ores.boxes:
            role = od_role_map.get(int(b.cls[0]))
            if role is not None:
                od_boxes.append((role, tuple(float(v) for v in b.xyxy[0])))

        # 3) person 별 GT / OD / MBC 판정
        for pid, pbox in enumerate(person_boxes, start=1):
            gt_helmet, gt_vest = make_gt(pbox, label_boxes, img_w, img_h)
            od_helmet, od_vest = od_predict(pbox, od_boxes, img_w, img_h)

            t0 = time.perf_counter()
            mbc_helmet, mbc_vest = mbc_predict(frame, pbox, mbc_model)
            timing["mbc"] += time.perf_counter() - t0

            # label 박스가 하나도 안 걸린 person = FP 검출 or 미라벨 가능성
            padded = expand_box(pbox, img_w, img_h, PERSON_BOX_PADDING)
            if not any(containment_ratio(b, padded) >= CONTAINMENT_RATIO
                       for _, b in label_boxes):
                diag["empty_person"] += 1

            records.append({
                "folder": name,
                "image": os.path.basename(img_path),
                "person_id": pid,
                "person_box": tuple(round(v, 1) for v in pbox),
                "gt_helmet": gt_helmet, "gt_vest": gt_vest,
                "od_helmet": od_helmet, "od_vest": od_vest,
                "mbc_helmet": mbc_helmet, "mbc_vest": mbc_vest,
            })

        timing["images"] += 1
        if i % 10 == 0 or i == len(image_files):
            print(f"  ...{i}/{len(image_files)} 처리")

    return records, timing, diag


# ==============================================================================
# 평가 지표
# ==============================================================================
def helmet_metrics(records, pipeline):
    """헬멧 정확도 + FNR. GT 가 None 인 person 은 평가 제외.
       OD 의 None 예측(검출 실패)은 오답으로 카운트."""
    key = f"{pipeline}_helmet"
    valid = [r for r in records if r["gt_helmet"] is not None]
    if not valid:
        return {"acc": None, "fnr": None, "n": 0, "n_neg": 0}
    correct = sum(1 for r in valid
                  if r[key] is not None and r[key] == r["gt_helmet"])
    neg = [r for r in valid if r["gt_helmet"] == 0]          # 실제 미착용
    fn = sum(1 for r in neg if r[key] == 1)                  # 미착용인데 착용으로 판정
    return {
        "acc": correct / len(valid),
        "fnr": (fn / len(neg)) if neg else None,
        "n": len(valid), "n_neg": len(neg),
    }


def vest_metrics(records, pipeline):
    """조끼 정확도 + FNR. GT/예측 모두 항상 0/1 이라 전 person 평가."""
    key = f"{pipeline}_vest"
    if not records:
        return {"acc": None, "fnr": None, "n": 0, "n_neg": 0}
    correct = sum(1 for r in records if r[key] == r["gt_vest"])
    neg = [r for r in records if r["gt_vest"] == 0]
    fn = sum(1 for r in neg if r[key] == 1)
    return {
        "acc": correct / len(records),
        "fnr": (fn / len(neg)) if neg else None,
        "n": len(records), "n_neg": len(neg),
    }

def binary_metrics_from_pairs(pairs):
    """
    pairs: [(gt, pred), ...], gt/pred는 0 또는 1
    1 = 착용, 0 = 미착용 기준.
    """
    if not pairs:
        return {
            "acc": None, "precision": None, "recall": None, "f1": None,
            "specificity": None, "false_safe_rate": None,
            "balanced_acc": None, "n": 0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }

    tp = sum(1 for gt, pred in pairs if gt == 1 and pred == 1)
    tn = sum(1 for gt, pred in pairs if gt == 0 and pred == 0)
    fp = sum(1 for gt, pred in pairs if gt == 0 and pred == 1)  # 미착용인데 착용으로 판단
    fn = sum(1 for gt, pred in pairs if gt == 1 and pred == 0)  # 착용인데 미착용으로 판단

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else None

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # 안전 관점 핵심: 실제 미착용인데 착용으로 판단한 비율
    false_safe_rate = fp / (tn + fp) if (tn + fp) else None

    if recall is not None and specificity is not None:
        balanced_acc = (recall + specificity) / 2
    else:
        balanced_acc = None

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_safe_rate": false_safe_rate,
        "balanced_acc": balanced_acc,
        "n": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def extended_metrics(records, pipeline, target):
    """
    target: 'helmet' 또는 'vest'
    pipeline: 'od' 또는 'mbc'
    """
    gkey = f"gt_{target}"
    pkey = f"{pipeline}_{target}"

    if target == "helmet":
        pairs = [(r[gkey], r[pkey]) for r in records if r[gkey] is not None]
    else:
        pairs = [(r[gkey], r[pkey]) for r in records]

    return binary_metrics_from_pairs(pairs)

def fmt(x):
    return "  N/A" if x is None else f"{x * 100:5.1f}%"


def print_summary(groups):
    """groups: [(라벨, records), ...]  — 폴더별 + 전체."""
    print("\n" + "=" * 72)
    print("성능 비교 요약  (OD 파이프라인 vs MBC 파이프라인)")
    print("=" * 72)
    for label, recs in groups:
        hm_od, hm_mbc = helmet_metrics(recs, "od"), helmet_metrics(recs, "mbc")
        vs_od, vs_mbc = vest_metrics(recs, "od"), vest_metrics(recs, "mbc")
        print(f"\n■ {label}   (person {len(recs)}명, "
              f"헬멧 평가대상 {hm_od['n']}명 / 제외 {len(recs) - hm_od['n']}명)")
        print(f"  {'':14s}{'헬멧 Acc':>10s}{'헬멧 FNR':>10s}"
              f"{'조끼 Acc':>10s}{'조끼 FNR':>10s}")
        print(f"  {'OD':14s}{fmt(hm_od['acc']):>10s}{fmt(hm_od['fnr']):>10s}"
              f"{fmt(vs_od['acc']):>10s}{fmt(vs_od['fnr']):>10s}")
        print(f"  {'MBC':14s}{fmt(hm_mbc['acc']):>10s}{fmt(hm_mbc['fnr']):>10s}"
              f"{fmt(vs_mbc['acc']):>10s}{fmt(vs_mbc['fnr']):>10s}")
    print("\n" + "-" * 72)
    print("FNR = 실제 미착용인데 착용으로 판정한 비율 (낮을수록 안전, 기획서 KPI ≤ 15%)")
    print("OD 헬멧: best.pt 가 head 미검출 -> helmet bbox 부재 = 미착용으로 추론.")
    print("GT 헬멧 None(helmet/no-helmet 라벨 모두 없음) 은 평가 제외.")
    print("=" * 72)


def print_timing(timing):
    imgs = max(timing["images"], 1)
    persons = max(timing.get("persons", 0), 1)

    person_name = os.path.basename(PERSON_MODEL_PATH)
    od_name = os.path.basename(OD_MODEL_PATH)
    mbc_name = os.path.basename(MBC_MODEL_PATH)

    person_ms_img = timing["person"] / imgs * 1000
    od_ms_img = timing["od"] / imgs * 1000
    mbc_ms_img = timing["mbc"] / imgs * 1000
    mbc_ms_person = timing["mbc"] / persons * 1000
    persons_per_img = timing.get("persons", 0) / imgs

    od_total_ms_img = person_ms_img + od_ms_img
    mbc_total_ms_img = person_ms_img + mbc_ms_img

    od_fps = 1000 / od_total_ms_img if od_total_ms_img > 0 else 0
    mbc_fps = 1000 / mbc_total_ms_img if mbc_total_ms_img > 0 else 0

    print(f"\n[추론 속도]  (PC 기준 — Pi5 실측 아님, 참고용)")
    print(f"  처리 이미지 수              : {timing['images']}장")
    print(f"  검출 person 수              : {timing.get('persons', 0)}명")
    print(f"  이미지당 평균 person 수      : {persons_per_img:6.2f} 명/img")
    print(f"  Person ({person_name})       : {person_ms_img:6.1f} ms/img")
    print(f"  OD ({od_name})               : {od_ms_img:6.1f} ms/img")
    print(f"  MBC ({mbc_name})             : {mbc_ms_img:6.1f} ms/img (이미지 내 전 인원 합)")
    print(f"  MBC 1인당 평균 처리 속도      : {mbc_ms_person:6.1f} ms/person")
    print(f"  OD 전체 파이프라인 추정       : {od_total_ms_img:6.1f} ms/img, {od_fps:5.2f} FPS")
    print(f"  MBC 전체 파이프라인 추정      : {mbc_total_ms_img:6.1f} ms/img, {mbc_fps:5.2f} FPS")


# ==============================================================================
# 시각화  — 2인 이하 / 3인 이상 폴더별로 구분
# ==============================================================================
def _safe_tag(name, idx):
    """폴더명을 파일명에 쓸 수 있게 정리. 'small (2인 이하)' -> 'small_2'."""
    t = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return t if t else f"folder{idx + 1}"


def save_confusion_matrices(per_folder, out_dir):
    """폴더(2인 이하 / 3인 이상)별로 혼동행렬 figure 를 1개씩 저장.
       각 figure = 2x2 (행: 헬멧/조끼, 열: OD/MBC). 1=착용, 0=미착용."""
    specs = [
        (0, 0, "Helmet - OD", "od_helmet", "gt_helmet", True),
        (0, 1, "Helmet - MBC", "mbc_helmet", "gt_helmet", True),
        (1, 0, "Vest - OD", "od_vest", "gt_vest", False),
        (1, 1, "Vest - MBC", "mbc_vest", "gt_vest", False),
    ]
    for idx, (fname, records) in enumerate(per_folder):
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for r, c, title, pkey, gkey, helmet in specs:
            ax = axes[r][c]
            if helmet:   # 헬멧은 GT None(판정불가) 제외
                pairs = [(rec[gkey], rec[pkey]) for rec in records
                         if rec[gkey] is not None and rec[pkey] is not None]
            else:
                pairs = [(rec[gkey], rec[pkey]) for rec in records]
            if not pairs:
                ax.set_title(title + " (no data)")
                ax.axis("off")
                continue
            cm = confusion_matrix([p[0] for p in pairs],
                                  [p[1] for p in pairs], labels=[1, 0])
            ConfusionMatrixDisplay(cm, display_labels=["Worn(1)", "Not(0)"]).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{title}  (n={len(pairs)})")
        fig.suptitle(f"Confusion Matrix — {fname}  (threshold={MBC_THRESHOLD})",
                     fontsize=14)
        fig.tight_layout()
        path = os.path.join(out_dir, f"confusion_matrix_{_safe_tag(fname, idx)}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  저장: {path}")


def save_metric_bars(per_folder, out_path):
    """2x2 — 상단 Accuracy(헬멧/조끼), 하단 FNR(헬멧/조끼).
       x축: 2인 이하 / 3인 이상 폴더,  막대: OD vs MBC.  (overall 은 표에만, 그래프엔 미포함)"""
    labels = [name for name, _ in per_folder]
    x = np.arange(len(labels))
    w = 0.36

    def collect(metric_fn, field):
        od = [(metric_fn(recs, "od")[field] or 0) for _, recs in per_folder]
        mbc = [(metric_fn(recs, "mbc")[field] or 0) for _, recs in per_folder]
        return od, mbc

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (axes[0][0], helmet_metrics, "acc", "Helmet Accuracy", KPI_HELMET_ACC),
        (axes[0][1], vest_metrics, "acc", "Vest Accuracy", KPI_VEST_ACC),
        (axes[1][0], helmet_metrics, "fnr", "Helmet FNR (lower=better)", KPI_FNR),
        (axes[1][1], vest_metrics, "fnr", "Vest FNR (lower=better)", KPI_FNR),
    ]
    for ax, fn, field, title, kpi in panels:
        od, mbc = collect(fn, field)
        b1 = ax.bar(x - w / 2, od, w, label="OD", color="#4C72B0")
        b2 = ax.bar(x + w / 2, mbc, w, label="MBC", color="#DD8452")
        ax.axhline(kpi, color="red", linestyle="--", linewidth=1,
                   label=f"KPI {kpi * 100:.0f}%")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{bar.get_height() * 100:.0f}", ha="center", fontsize=9)
    fig.suptitle("OD vs MBC  —  2인 이하 / 3인 이상 비교", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  저장: {out_path}")


def save_summary_csv(groups, out_path):
    """폴더별 + 전체 성능 지표 요약표 — 슬라이드/엑셀 붙여넣기용."""
    cols = [
        "folder", "pipeline",

        "helmet_acc", "helmet_precision", "helmet_recall", "helmet_f1",
        "helmet_specificity", "helmet_false_safe_rate", "helmet_balanced_acc",
        "helmet_n", "helmet_tp", "helmet_tn", "helmet_fp", "helmet_fn",

        "vest_acc", "vest_precision", "vest_recall", "vest_f1",
        "vest_specificity", "vest_false_safe_rate", "vest_balanced_acc",
        "vest_n", "vest_tp", "vest_tn", "vest_fp", "vest_fn",

        "person_exact_match", "person_exact_match_n",
    ]

    def cell(v):
        return "" if v is None else round(v, 4)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()

        for label, recs in groups:
            for pipe in ("od", "mbc"):
                hm = extended_metrics(recs, pipe, "helmet")
                vm = extended_metrics(recs, pipe, "vest")
                em = ppe_exact_match_metrics(recs, pipe)

                wr.writerow({
                    "folder": label,
                    "pipeline": pipe.upper(),

                    "helmet_acc": cell(hm["acc"]),
                    "helmet_precision": cell(hm["precision"]),
                    "helmet_recall": cell(hm["recall"]),
                    "helmet_f1": cell(hm["f1"]),
                    "helmet_specificity": cell(hm["specificity"]),
                    "helmet_false_safe_rate": cell(hm["false_safe_rate"]),
                    "helmet_balanced_acc": cell(hm["balanced_acc"]),
                    "helmet_n": hm["n"],
                    "helmet_tp": hm["tp"],
                    "helmet_tn": hm["tn"],
                    "helmet_fp": hm["fp"],
                    "helmet_fn": hm["fn"],

                    "vest_acc": cell(vm["acc"]),
                    "vest_precision": cell(vm["precision"]),
                    "vest_recall": cell(vm["recall"]),
                    "vest_f1": cell(vm["f1"]),
                    "vest_specificity": cell(vm["specificity"]),
                    "vest_false_safe_rate": cell(vm["false_safe_rate"]),
                    "vest_balanced_acc": cell(vm["balanced_acc"]),
                    "vest_n": vm["n"],
                    "vest_tp": vm["tp"],
                    "vest_tn": vm["tn"],
                    "vest_fp": vm["fp"],
                    "vest_fn": vm["fn"],

                    "person_exact_match": cell(em["person_exact_match"]),
                    "person_exact_match_n": em["n"],
                })

    print(f"  저장: {out_path}")


def save_csv(records, out_path):
    """per-person 원자료 — 발표 백업용."""
    cols = ["folder", "image", "person_id", "person_box",
            "gt_helmet", "od_helmet", "mbc_helmet",
            "gt_vest", "od_vest", "mbc_vest"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in records:
            wr.writerow(r)
    print(f"  저장: {out_path}")

def ppe_exact_match_metrics(records, pipeline):
    """
    사람 1명 기준으로 헬멧/조끼를 모두 맞춘 비율.
    헬멧 GT가 None인 경우는 헬멧 비교를 제외하고 조끼만 비교.
    """
    if not records:
        return {"person_exact_match": None, "n": 0}

    correct = 0
    valid_count = 0

    for r in records:
        vest_ok = r[f"{pipeline}_vest"] == r["gt_vest"]

        if r["gt_helmet"] is None:
            # 헬멧 라벨이 없으면 조끼만 평가
            helmet_ok = True
        else:
            helmet_ok = r[f"{pipeline}_helmet"] == r["gt_helmet"]

        if helmet_ok and vest_ok:
            correct += 1
        valid_count += 1

    return {
        "person_exact_match": correct / valid_count if valid_count else None,
        "n": valid_count,
    }

def print_extended_summary(groups):
    print("\n" + "=" * 72)
    print("확장 성능 지표 요약")
    print("=" * 72)

    for label, recs in groups:
        print(f"\n■ {label}   (person {len(recs)}명)")
        print(f"  {'':10s}{'대상':8s}{'Acc':>8s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}"
              f"{'Spec':>8s}{'FalseSafe':>11s}{'BalAcc':>8s}{'Exact':>8s}")

        for pipe in ("od", "mbc"):
            hm = extended_metrics(recs, pipe, "helmet")
            vm = extended_metrics(recs, pipe, "vest")
            em = ppe_exact_match_metrics(recs, pipe)

            print(f"  {pipe.upper():10s}{'Helmet':8s}"
                  f"{fmt(hm['acc']):>8s}"
                  f"{fmt(hm['precision']):>8s}"
                  f"{fmt(hm['recall']):>8s}"
                  f"{fmt(hm['f1']):>8s}"
                  f"{fmt(hm['specificity']):>8s}"
                  f"{fmt(hm['false_safe_rate']):>11s}"
                  f"{fmt(hm['balanced_acc']):>8s}"
                  f"{'':>8s}")

            print(f"  {pipe.upper():10s}{'Vest':8s}"
                  f"{fmt(vm['acc']):>8s}"
                  f"{fmt(vm['precision']):>8s}"
                  f"{fmt(vm['recall']):>8s}"
                  f"{fmt(vm['f1']):>8s}"
                  f"{fmt(vm['specificity']):>8s}"
                  f"{fmt(vm['false_safe_rate']):>11s}"
                  f"{fmt(vm['balanced_acc']):>8s}"
                  f"{fmt(em['person_exact_match']):>8s}")

    print("\nFalseSafe = 실제 미착용인데 착용으로 판단한 비율")
    print("Exact = 사람 1명 기준 헬멧/조끼 상태를 모두 맞춘 비율")
    print("=" * 72)

# ==============================================================================
# 메인
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("모델 로딩 중...")
    person_model = YOLO(PERSON_MODEL_PATH)
    od_model = YOLO(OD_MODEL_PATH)
    od_role_map = build_od_class_map(od_model)
    print(f"  OD 모델 클래스: {od_model.names}")
    print(f"  OD 역할 매핑  : {od_role_map}   (head = no-helmet 취급)")
    if not od_role_map:
        raise SystemExit("OD 모델에서 helmet/head/vest 클래스를 찾지 못했습니다. "
                         "build_od_class_map 의 이름 매핑을 확인하세요.")
    mbc_model = MBCModel(MBC_MODEL_PATH)

    all_records = []
    all_timing = {"person": 0.0, "od": 0.0, "mbc": 0.0, "images": 0, "persons": 0}
    all_diag = {"no_person": 0, "no_label": 0, "empty_person": 0}
    per_folder = []

    for name, images_dir, labels_dir in DATASETS:
        recs, timing, diag = process_dataset(
            name, images_dir, labels_dir,
            person_model, od_model, od_role_map, mbc_model)

        per_folder.append((name, recs))
        all_records.extend(recs)

        for k in all_timing:
            all_timing[k] += timing[k]

        for k in all_diag:
            all_diag[k] += diag[k]

    if not all_records:
        raise SystemExit("처리된 person 레코드가 없습니다. 경로 설정을 확인하세요.")

    groups = per_folder + [("전체 (overall)", all_records)]

    print_summary(groups)
    print_extended_summary(groups)
    print_timing(all_timing)

    print(f"\n[진단]")
    print(f"  사람 미검출 이미지        : {all_diag['no_person']}장")
    print(f"  라벨 파일 없는 이미지     : {all_diag['no_label']}장")
    print(f"  라벨 박스 0개 귀속 person : {all_diag['empty_person']}명 "
          f"(FP 검출 or 미라벨 가능성 — 많으면 확인 필요)")

    print(f"\n[결과물 저장]  -> {OUTPUT_DIR}")
    save_metric_bars(per_folder, os.path.join(OUTPUT_DIR, "metrics_bar.png"))
    save_confusion_matrices(per_folder, OUTPUT_DIR)
    save_summary_csv(groups, os.path.join(OUTPUT_DIR, "summary_metrics.csv"))
    save_csv(all_records, os.path.join(OUTPUT_DIR, "per_person_results.csv"))
    print("\n완료.")
    
if __name__ == "__main__":
    main()
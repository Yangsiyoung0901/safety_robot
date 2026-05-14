"""
================================================================================
크롭 디버그 스크립트 — MBC 입력 크롭을 파일로 저장해서 눈으로 확인
================================================================================
ppe_eval_compare.py 와 *완전히 동일한* 크롭 로직으로
  yolo11m person detection (conf 0.25 + IoU 중복 제거)
  -> crop_upper_body_box (팀원 C 의 crop_upper_body 와 동일)
  -> resize_with_padding(224)
를 수행하고, 그 결과(= MBC 모델이 실제로 보는 입력)를 파일로 저장한다.

저장 구조:
  <OUT_DIR>/
    helmet_1/    GT 헬멧 착용 인 사람들의 크롭
    helmet_0/    GT 헬멧 미착용 인 사람들의 크롭
    helmet_none/ GT 헬멧 판정불가(helmet/no-helmet 라벨 모두 없음)
  파일명: {folder}__{이미지명}__p{사람ID}__v{GT조끼}.jpg

사용법:
  helmet_0 폴더를 열어서 — 정말 헬멧을 안 쓴 사람 크롭인지,
  헬멧이 잘려나갔거나 옆 사람 헬멧이 들어와 있지 않은지 확인.
  train_sampled(학습 크롭)와 나란히 비교해서 크롭 형태가 같은지 본다.
================================================================================
"""

import os
import glob

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================================================================
# ★ 설정 — ppe_eval_compare.py 와 동일하게 유지할 것
# ==============================================================================
PERSON_MODEL_PATH = "yolo11m.pt"     # MBC 학습 크롭과 동일 detector

DATASETS = [
    ("small",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\small\images",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\small\labels"),
    ("large",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\large\images",
     r"D:\AIProject\YSY\gt_test_seperate\gt_test\originals\large\labels"),
]

OUT_DIR = r"D:\AIProject\YSY\eval_results\crop_debug"

# 아래 값은 ppe_eval_compare.py 와 반드시 동일해야 함 (팀원 C Crop_with_labels.ipynb 기준)
PERSON_CONF        = 0.25
PERSON_IMGSZ       = 640
CONTAINMENT_RATIO  = 0.5
PERSON_BOX_PADDING = 0.10
MBC_INPUT_SIZE     = 224
IOU_DUPLICATE_THRESHOLD = 0.5
UPPER_BODY_RATIO   = 0.65

LABEL_HELMET    = 0
LABEL_VEST      = 1
LABEL_NO_HELMET = 2

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


# ==============================================================================
# 헬퍼 — ppe_eval_compare.py 에서 그대로 가져옴 (수정하지 말 것)
# ==============================================================================
def containment_ratio(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    iw = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    ih = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inter = iw * ih
    inner_area = max((ix2 - ix1) * (iy2 - iy1), 1e-6)
    return inter / inner_area


def expand_box(box, img_w, img_h, ratio):
    x1, y1, x2, y2 = box
    px = (x2 - x1) * ratio
    py = (y2 - y1) * ratio
    return (max(0.0, x1 - px), max(0.0, y1 - py),
            min(img_w - 1.0, x2 + px), min(img_h - 1.0, y2 + py))


def parse_label_file(path, img_w, img_h):
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
    """IoU 가 높은 중복 person bbox 제거. persons: [{'box':..., 'conf':...}, ...]"""
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
    """팀원 C 의 crop_upper_body 와 동일: 종횡비 분기, 좌우/상단 패딩 없음."""
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
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def make_gt(person_box, label_boxes, img_w, img_h):
    padded = expand_box(person_box, img_w, img_h, PERSON_BOX_PADDING)
    helmet = [b for c, b in label_boxes if c == LABEL_HELMET]
    nohelm = [b for c, b in label_boxes if c == LABEL_NO_HELMET]
    vest = [b for c, b in label_boxes if c == LABEL_VEST]
    has_helmet = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in helmet)
    has_nohelm = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in nohelm)
    has_vest = any(containment_ratio(b, padded) >= CONTAINMENT_RATIO for b in vest)
    if has_helmet:
        gt_helmet = 1
    elif has_nohelm:
        gt_helmet = 0
    else:
        gt_helmet = None
    gt_vest = 1 if has_vest else 0
    return gt_helmet, gt_vest


# ==============================================================================
# 메인
# ==============================================================================
def main():
    sub = {1: "helmet_1", 0: "helmet_0", None: "helmet_none"}
    for d in sub.values():
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    print("yolo11m 로딩...")
    model = YOLO(PERSON_MODEL_PATH)

    counts = {1: 0, 0: 0, None: 0}
    for tag, images_dir, labels_dir in DATASETS:
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(images_dir, ext)))
        files = sorted(files)
        print(f"\n[{tag}]  이미지 {len(files)}장")
        if not files:
            print(f"  ⚠️  경로 확인: {images_dir}")
            continue

        for i, img_path in enumerate(files, start=1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label_boxes = parse_label_file(
                os.path.join(labels_dir, stem + ".txt"), img_w, img_h)

            res = model.predict(source=frame, imgsz=PERSON_IMGSZ,
                                conf=PERSON_CONF, classes=[0], verbose=False)[0]
            persons = [{"box": tuple(float(v) for v in b.xyxy[0]),
                        "conf": float(b.conf[0])} for b in res.boxes]
            persons = remove_duplicate_boxes(persons)
            boxes = [p["box"] for p in persons]
            boxes.sort(key=lambda b: b[0])    # 좌->우, ppe_eval_compare.py 와 동일

            for pid, pbox in enumerate(boxes, start=1):
                gt_h, gt_v = make_gt(pbox, label_boxes, img_w, img_h)

                crop_box = crop_upper_body_box(pbox, img_w, img_h)
                if crop_box is None:
                    crop_box = pbox          # 퇴화 시 person 박스 전체로 폴백
                x1, y1, x2, y2 = (int(round(v)) for v in crop_box)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue

                # MBC 가 보는 입력과 동일한 패딩 (색상은 보기 좋게 BGR 유지)
                padded = resize_with_padding(crop, MBC_INPUT_SIZE)

                fname = f"{tag}__{stem}__p{pid}__v{gt_v}.jpg"
                cv2.imwrite(os.path.join(OUT_DIR, sub[gt_h], fname), padded)
                counts[gt_h] += 1

            if i % 10 == 0 or i == len(files):
                print(f"  ...{i}/{len(files)}")

    total = sum(counts.values())
    print(f"\n완료: 총 {total}개 크롭 저장 -> {OUT_DIR}")
    print(f"  helmet_1   (착용)     : {counts[1]}개")
    print(f"  helmet_0   (미착용)   : {counts[0]}개")
    print(f"  helmet_none(판정불가) : {counts[None]}개")
    print("\n확인 포인트:")
    print("  - helmet_0 폴더: 정말 헬멧 안 쓴 사람인가? 헬멧이 잘려나가거나")
    print("                   옆 사람 헬멧이 크롭에 들어와 있지 않은가?")
    print("  - train_sampled(학습 크롭)와 나란히 놓고 크롭 형태/비율이 같은지 비교")


if __name__ == "__main__":
    main()

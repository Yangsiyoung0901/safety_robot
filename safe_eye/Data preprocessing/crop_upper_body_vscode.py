# -*- coding: utf-8 -*-
"""
YOLO 사람 감지 → 상체 크롭 → 원본 라벨 매칭 코드 (VSCode / 로컬 환경용)

기능:
  1. YOLO로 이미지에서 사람을 감지하고 상체를 크롭
  2. 원본 YOLO 라벨(.txt)에서 크롭 영역에 포함되는 라벨만 추출
  3. 라벨 좌표를 크롭 이미지 기준으로 변환하여 새 라벨 파일로 저장

출력 구조:
  CROP_OUTPUT_DIR/
  ├── images/
  │   ├── 000500_person1_upper.jpg
  │   └── ...
  └── labels/
      ├── 000500_person1_upper.txt   ← 크롭 기준 재계산된 YOLO 라벨
      └── ...

실행 전 설치:
    pip install ultralytics opencv-python matplotlib numpy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =============================================================
# ★ 경로 설정 — 본인 환경에 맞게 수정하세요
# =============================================================

# 이미지 폴더 경로
IMAGE_DIR = r"C:\Users\KCCISTC\Documents\CropSampleImage\Images"

# 라벨 폴더 경로 (YOLO .txt 파일)
LABEL_DIR = r"C:\Users\KCCISTC\Documents\CropSampleImage\Labels"

# 크롭 결과 저장 폴더
CROP_OUTPUT_DIR = r"C:\Users\KCCISTC\Documents\cropped_upper_body"

os.makedirs(os.path.join(CROP_OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(CROP_OUTPUT_DIR, "labels"), exist_ok=True)

# =============================================================
# 설정값
# =============================================================

UPPER_BODY_RATIO = 0.65       # 전신 bbox에서 상체로 간주할 비율 (상위 65%)
PERSON_CONFIDENCE = 0.25      # 사람 감지 신뢰도 임계값
MIN_CROP_SIZE = 50            # 크롭 최소 크기 (px)
LABEL_CONTAIN_RATIO = 0.5     # 라벨 bbox가 크롭 영역에 이 비율 이상 포함되어야 매칭

print(f"이미지 폴더: {IMAGE_DIR}")
print(f"라벨 폴더:   {LABEL_DIR}")
print(f"크롭 저장:   {CROP_OUTPUT_DIR}")
print(f"상체 비율:   상위 {UPPER_BODY_RATIO * 100:.0f}%")
print(f"라벨 포함 기준: {LABEL_CONTAIN_RATIO * 100:.0f}%")
print()

# =============================================================
# 모델 로드
# =============================================================

model = YOLO("yolo11m.pt")  # 자동 다운로드됨
print("모델 로드 완료\n")

# =============================================================
# 핵심 함수
# =============================================================

def calculate_iou(box1, box2):
    """두 bbox의 IoU를 계산합니다. (x1, y1, x2, y2) 형식"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def remove_duplicate_boxes(persons, iou_threshold=0.5):
    """IoU가 높은 중복 bbox를 제거합니다."""
    if len(persons) <= 1:
        return persons

    persons = sorted(persons, key=lambda p: p["confidence"], reverse=True)
    keep = []

    for person in persons:
        is_duplicate = False
        for kept in keep:
            if calculate_iou(person["bbox"], kept["bbox"]) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(person)

    return keep


def detect_persons(model, image):
    """YOLO로 사람(class 0)을 감지하고 bbox 목록을 반환합니다."""
    results = model(image, verbose=False)
    persons = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == 0 and conf >= PERSON_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })

    persons = remove_duplicate_boxes(persons, iou_threshold=0.5)
    return persons


def crop_upper_body(image, bbox, ratio=UPPER_BODY_RATIO):
    """person bbox에서 상체 영역만 크롭합니다."""
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    width = x2 - x1
    aspect_ratio = height / width if width > 0 else 0

    if aspect_ratio >= 2.0:
        upper_y2 = y1 + int(height * ratio)
    else:
        if aspect_ratio < 1.0:
            extend = int(height * 0.5)
            upper_y2 = y2 + extend
        else:
            upper_y2 = y2

    h, w = image.shape[:2]
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, upper_y2)

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)

    return cropped, crop_bbox


# =============================================================
# 라벨 매칭 함수
# =============================================================

def load_yolo_labels(label_path, img_w, img_h):
    """
    YOLO 라벨 파일을 읽어서 픽셀 좌표로 변환합니다.

    Returns:
        list of dict: [{"class_id": int, "bbox_pixel": (x1, y1, x2, y2),
                         "original_line": str}, ...]
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            bw = float(parts[3]) * img_w
            bh = float(parts[4]) * img_h

            lx1 = cx - bw / 2
            ly1 = cy - bh / 2
            lx2 = cx + bw / 2
            ly2 = cy + bh / 2

            labels.append({
                "class_id": cls_id,
                "bbox_pixel": (lx1, ly1, lx2, ly2),
                "original_line": line
            })

    return labels


def calculate_contain_ratio(label_bbox, crop_bbox):
    """
    라벨 bbox가 크롭 영역에 얼마나 포함되는지 비율을 계산합니다.
    (라벨 bbox 면적 중 크롭 영역 내에 들어가는 비율)
    """
    lx1, ly1, lx2, ly2 = label_bbox
    cx1, cy1, cx2, cy2 = crop_bbox

    ix1 = max(lx1, cx1)
    iy1 = max(ly1, cy1)
    ix2 = min(lx2, cx2)
    iy2 = min(ly2, cy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    label_area = (lx2 - lx1) * (ly2 - ly1)

    return inter / label_area if label_area > 0 else 0


def match_labels_to_crop(labels, crop_bbox, crop_w, crop_h, contain_ratio=LABEL_CONTAIN_RATIO):
    """
    원본 라벨 중 크롭 영역에 포함되는 라벨을 찾아
    크롭 이미지 기준 YOLO 좌표로 변환합니다.
    """
    cx1, cy1, cx2, cy2 = crop_bbox
    matched_lines = []

    for label in labels:
        ratio = calculate_contain_ratio(label["bbox_pixel"], crop_bbox)

        if ratio >= contain_ratio:
            lx1, ly1, lx2, ly2 = label["bbox_pixel"]

            new_x1 = max(lx1, cx1) - cx1
            new_y1 = max(ly1, cy1) - cy1
            new_x2 = min(lx2, cx2) - cx1
            new_y2 = min(ly2, cy2) - cy1

            if crop_w > 0 and crop_h > 0:
                new_cx = ((new_x1 + new_x2) / 2) / crop_w
                new_cy = ((new_y1 + new_y2) / 2) / crop_h
                new_bw = (new_x2 - new_x1) / crop_w
                new_bh = (new_y2 - new_y1) / crop_h

                new_cx = max(0.0, min(1.0, new_cx))
                new_cy = max(0.0, min(1.0, new_cy))
                new_bw = max(0.0, min(1.0, new_bw))
                new_bh = max(0.0, min(1.0, new_bh))

                if new_bw > 0.01 and new_bh > 0.01:
                    line = f"{label['class_id']} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}"
                    matched_lines.append(line)

    return matched_lines


# =============================================================
# 미리보기 함수 (matplotlib 팝업 창으로 표시)
# =============================================================

def preview_crops(image_path, model, max_persons=3):
    """
    한 장의 이미지에 대해 감지 결과와 크롭 영역을 시각화합니다.
    크롭 이미지 위에 매칭된 라벨 bbox를 표시하고, 라벨 텍스트도 출력합니다.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return

    img_h, img_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    persons = detect_persons(model, image_rgb)

    # 원본 라벨 로드
    label_fname = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_fname)
    labels = load_yolo_labels(label_path, img_w, img_h)

    if not persons:
        print("감지된 사람이 없습니다.")
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)
        plt.title("No person detected")
        plt.axis("off")
        plt.show()
        return

    n = min(len(persons), max_persons)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    if n + 1 == 1:
        axes = [axes]

    # 원본 이미지에 person bbox + 원본 라벨 bbox 표시
    display = image_rgb.copy()
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f'person {p["confidence"]:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for lb in labels:
        lx1, ly1, lx2, ly2 = map(int, lb["bbox_pixel"])
        cv2.rectangle(display, (lx1, ly1), (lx2, ly2), (255, 0, 0), 1)
        cv2.putText(display, f'cls={lb["class_id"]}', (lx1, ly1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    axes[0].imshow(display)
    axes[0].set_title(f"Original ({len(persons)} persons, {len(labels)} labels)")
    axes[0].axis("off")

    # 각 사람의 상체 크롭 + 매칭 라벨 시각화
    for i in range(n):
        cropped, crop_bbox = crop_upper_body(image_rgb, persons[i]["bbox"])
        crop_h, crop_w = cropped.shape[:2]

        # 매칭된 라벨 구하기
        matched_labels = match_labels_to_crop(labels, crop_bbox, crop_w, crop_h)

        # 크롭 이미지 위에 매칭된 라벨 bbox 그리기
        crop_display = cropped.copy()
        for line in matched_labels:
            parts = line.split()
            cls_id = int(parts[0])
            cx_n, cy_n, bw_n, bh_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bx1 = int((cx_n - bw_n / 2) * crop_w)
            by1 = int((cy_n - bh_n / 2) * crop_h)
            bx2 = int((cx_n + bw_n / 2) * crop_w)
            by2 = int((cy_n + bh_n / 2) * crop_h)
            cv2.rectangle(crop_display, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.putText(crop_display, f'cls={cls_id}', (bx1, by1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        axes[i + 1].imshow(crop_display)
        axes[i + 1].set_title(f"Person {i+1} crop ({len(matched_labels)} labels)")
        axes[i + 1].axis("off")

        # 라벨 텍스트 출력
        print(f"\n📌 Person {i+1} 매칭 라벨 ({len(matched_labels)}개):")
        if matched_labels:
            for j, line in enumerate(matched_labels):
                print(f"   [{j+1}] {line}")
        else:
            print(f"   (매칭된 라벨 없음)")

    plt.tight_layout()
    plt.show()


# =============================================================
# 메인 처리 함수
# =============================================================

def process_image(model, image_path, label_path):
    """
    한 장의 이미지를 처리합니다:
    1) YOLO로 사람 감지
    2) 각 사람의 상체 크롭
    3) 원본 라벨에서 크롭 영역에 포함되는 라벨 매칭 & 좌표 변환
    4) 크롭 이미지 + 변환된 라벨 파일 저장

    Returns:
        int: 저장된 크롭 수
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ⛔ 이미지를 읽을 수 없습니다: {image_path}")
        return 0

    img_h, img_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 원본 라벨 로드
    labels = load_yolo_labels(label_path, img_w, img_h)

    # 1) 사람 감지
    persons = detect_persons(model, image_rgb)

    if not persons:
        print(f"  감지된 사람: 0명")
        return 0

    print(f"  감지된 사람: {len(persons)}명 | 원본 라벨: {len(labels)}개")

    filename = os.path.basename(image_path)
    name_only = os.path.splitext(filename)[0]
    saved_count = 0

    for i, person in enumerate(persons):
        bbox = person["bbox"]

        # 2) 상체 크롭
        cropped, crop_bbox = crop_upper_body(image_rgb, bbox)
        crop_h, crop_w = cropped.shape[:2]

        # 크롭 최소 크기 필터
        if crop_h < MIN_CROP_SIZE or crop_w < MIN_CROP_SIZE:
            print(f"    ⛔ 사람 {i+1} 크롭 제외 ({crop_w}×{crop_h}px — 너무 작음)")
            continue

        # 3) 라벨 매칭 & 좌표 변환
        matched_labels = match_labels_to_crop(labels, crop_bbox, crop_w, crop_h)

        # 4) 클래스 중복 검사 — 같은 클래스가 2개 이상이면 저장 스킵
        class_ids = [int(line.split()[0]) for line in matched_labels]
        has_duplicate = len(class_ids) != len(set(class_ids))

        person_bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"

        if has_duplicate:
            print(f"    ⛔ 사람 {i+1} 저장 제외 | bbox={person_bbox_str} | "
                  f"클래스 중복 발생: {class_ids}")
            for j, line in enumerate(matched_labels):
                print(f"       라벨 [{j+1}] {line}")
            continue

        # 5) 헬멧·조끼 착용 여부 코드 생성
        # 클래스 0: 헬멧O, 1: 조끼O, 2: 헬멧X, 3: 조끼X
        # 파일명 코드: 첫째 자리=헬멧(1:O, 0:X), 둘째 자리=조끼(1:O, 0:X)
        helmet_code = "1" if 0 in class_ids else "0"  # 클래스 0 있으면 헬멧O
        vest_code = "1" if 1 in class_ids else "0"    # 클래스 1 있으면 조끼O
        status_code = helmet_code + vest_code  # 예: "11", "10", "01", "00"

        # 6) 저장
        crop_name = f"{name_only}_person{i+1}_{status_code}"

        save_img_path = os.path.join(CROP_OUTPUT_DIR, "images", f"{crop_name}.jpg")
        cv2.imwrite(save_img_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        save_label_path = os.path.join(CROP_OUTPUT_DIR, "labels", f"{crop_name}.txt")
        with open(save_label_path, "w") as f:
            for line in matched_labels:
                f.write(line + "\n")

        status_desc = f"헬멧={'O' if helmet_code=='1' else 'X'} 조끼={'O' if vest_code=='1' else 'X'}"
        print(f"    ✅ 사람 {i+1} | bbox={person_bbox_str} | "
              f"크롭={crop_w}×{crop_h}px | {status_desc} | 매칭 라벨={len(matched_labels)}개")

        # 매칭된 라벨 텍스트 출력
        for j, line in enumerate(matched_labels):
            print(f"       라벨 [{j+1}] {line}")

        saved_count += 1

    return saved_count


# =============================================================
# 실행
# =============================================================

if __name__ == "__main__":
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in EXTENSIONS
    ])

    print(f"처리할 이미지: {len(image_files)}장\n")

    # ★ 미리보기: 첫 번째 이미지로 결과 확인 (선택사항, 창을 닫으면 전체 처리 진행)
    if image_files:
        print("=" * 50)
        print("🔍 첫 번째 이미지 미리보기")
        print("=" * 50)
        preview_crops(os.path.join(IMAGE_DIR, image_files[0]), model)
        print()

    # 전체 처리
    total_crops = 0

    for idx, fname in enumerate(image_files):
        print(f"[{idx+1}/{len(image_files)}] {fname}")

        img_path = os.path.join(IMAGE_DIR, fname)
        label_fname = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_fname)

        if not os.path.exists(label_path):
            print(f"  ⚠️ 라벨 파일 없음: {label_fname}")

        saved = process_image(model, img_path, label_path)
        total_crops += saved

    print(f"\n{'='*50}")
    print(f"✅ 완료!")
    print(f"총 크롭 이미지: {total_crops}장")
    print(f"저장 위치: {CROP_OUTPUT_DIR}")
    print(f"  ├── images/   ← 크롭된 이미지")
    print(f"  └── labels/   ← 변환된 YOLO 라벨")

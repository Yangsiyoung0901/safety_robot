"""
filter_single_person_colab.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO11m으로 원본 사진에서 person이 정확히 1명인 사진만 골라내고,
원본 YOLO 라벨(0=helmet, 1=vest, 2=no-helmet, 3=no-vest)과 매칭하여
파일명에 HV 라벨을 붙여 새 폴더에 복사합니다.

출력 파일명: {split}_{원본이름}_{HV}.jpg
  H: 1=helmet 착용, 0=미착용
  V: 1=vest 착용,   0=미착용

Colab에서 # %% 구분자로 셀별 실행하세요.
"""

# %%
# ═══════════════════════════════════════════
# [셀 1] 설치 & Drive 마운트
# ═══════════════════════════════════════════

# !pip install ultralytics

# from google.colab import drive
# drive.mount('/content/drive')

# %%
# ═══════════════════════════════════════════
# [셀 2] 경로 & 설정 ★ 수정 ★
# ═══════════════════════════════════════════

import os
import shutil
from ultralytics import YOLO

# ★ 본인 환경에 맞게 수정 ★
BASE_DIR = "/content/drive/MyDrive/Safe/archive_00"
SPLITS = ["train", "val", "test"]
OUTPUT_DIR = "/content/drive/MyDrive/Safe/single_person_filtered"

PERSON_CONFIDENCE = 0.25
IOU_DUPLICATE_THRESHOLD = 0.5

# 원본 라벨 클래스 ID
# 0=helmet, 1=vest, 2=no-helmet, 3=no-vest
HELMET_ID = 0
VEST_ID = 1
NO_HELMET_ID = 2
NO_VEST_ID = 3

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

print(f"원본 경로: {BASE_DIR}")
print(f"저장 경로: {OUTPUT_DIR}")
print(f"처리 대상: {SPLITS}\n")

# %%
# ═══════════════════════════════════════════
# [셀 3] 모델 로드
# ═══════════════════════════════════════════

model = YOLO("yolo11m.pt")
print("모델 로드 완료\n")

# %%
# ═══════════════════════════════════════════
# [셀 4] 핵심 함수
# ═══════════════════════════════════════════

def calculate_iou(box1, box2):
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
    if len(persons) <= 1:
        return persons
    keep = []
    used = set()
    sorted_p = sorted(persons, key=lambda p: p['conf'], reverse=True)
    for i, p in enumerate(sorted_p):
        if i in used:
            continue
        keep.append(p)
        for j in range(i + 1, len(sorted_p)):
            if j not in used and calculate_iou(p['bbox'], sorted_p[j]['bbox']) > iou_threshold:
                used.add(j)
    return keep


def get_wear_code(label_path):
    """
    원본 YOLO 라벨 파일에서 HV 코드 결정.
    H: 1 if helmet(0) 존재, else 0
    V: 1 if vest(1) 존재,   else 0
    """
    if not os.path.exists(label_path):
        return None

    class_ids = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.add(int(parts[0]))

    if not class_ids:
        return None

    h = '1' if HELMET_ID in class_ids else '0'
    v = '1' if VEST_ID in class_ids else '0'
    return h + v


# %%
# ═══════════════════════════════════════════
# [셀 5] 실행
# ═══════════════════════════════════════════

total_stats = {'total': 0, 'single': 0, 'multi': 0, 'no_person': 0, 'no_label': 0}
code_counts = {'00': 0, '01': 0, '10': 0, '11': 0}

# 이어서 하기용: 이미 처리된 파일 확인
existing = set(os.listdir(os.path.join(OUTPUT_DIR, "images")))
print(f"기존 파일: {len(existing)}개 (이미 처리된 원본 건너뜁니다)\n")

for split in SPLITS:
    image_dir = os.path.join(BASE_DIR, "images", split)
    label_dir = os.path.join(BASE_DIR, "labels", split)

    if not os.path.exists(image_dir):
        print(f"[{split}] 폴더 없음: {image_dir}")
        continue

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"[{split}] {len(image_files)}장 처리 시작...")
    split_single = 0

    for idx, fname in enumerate(image_files):
        total_stats['total'] += 1
        base_name = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]

        # 이미 처리됨 → 건너뛰기
        if any(f.startswith(f"{split}_{base_name}_") for f in existing):
            continue

        img_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, base_name + '.txt')

        # YOLO11m person 감지
        results = model(img_path, conf=PERSON_CONFIDENCE, verbose=False)
        persons = []
        for box in results[0].boxes:
            if int(box.cls[0]) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                persons.append({'bbox': (x1, y1, x2, y2), 'conf': float(box.conf[0])})
        persons = remove_duplicate_boxes(persons)

        if len(persons) == 0:
            total_stats['no_person'] += 1
            continue
        if len(persons) > 1:
            total_stats['multi'] += 1
            continue

        # 1명 확인 → 라벨 매칭
        code = get_wear_code(label_path)
        if code is None:
            total_stats['no_label'] += 1
            continue

        # 원본 이미지 복사 (파일명에 라벨 포함)
        out_name = f"{split}_{base_name}_{code}{ext}"
        shutil.copy2(img_path, os.path.join(OUTPUT_DIR, "images", out_name))

        # classification 라벨 저장
        lbl_out = os.path.join(OUTPUT_DIR, "labels", f"{split}_{base_name}_{code}.txt")
        with open(lbl_out, 'w') as f:
            f.write(code + '\n')

        total_stats['single'] += 1
        code_counts[code] += 1
        split_single += 1

        # 진행률 (200장마다)
        if (idx + 1) % 200 == 0:
            print(f"  [{split}] {idx+1}/{len(image_files)} | 1인: {split_single}")

    print(f"  [{split}] 완료 → 1인 사진: {split_single}장\n")

# 결과 출력
print("=" * 50)
print("전체 결과")
print("=" * 50)
print(f"  총 이미지:       {total_stats['total']}")
print(f"  1인 사진 (저장): {total_stats['single']}")
print(f"  다인 제외:       {total_stats['multi']}")
print(f"  미감지 제외:     {total_stats['no_person']}")
print(f"  라벨없음 제외:   {total_stats['no_label']}")
print()
print("  라벨 분포:")
print(f"    11 (헬멧O 조끼O): {code_counts['11']}")
print(f"    10 (헬멧O 조끼X): {code_counts['10']}")
print(f"    01 (헬멧X 조끼O): {code_counts['01']}")
print(f"    00 (헬멧X 조끼X): {code_counts['00']}")
print()
print(f"  저장 위치: {OUTPUT_DIR}")
print(f"    ├── images/  (원본 사진 복사)")
print(f"    └── labels/  (HV 코드: 00/01/10/11)")
print("=" * 50)

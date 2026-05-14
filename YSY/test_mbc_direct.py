"""
================================================================================
MBC 직접 테스트 — train_sampled 크롭을 yolo12n/크롭 없이 모델에 바로 투입
================================================================================
목적: "MBC 헬멧이 안 맞는다"의 원인이
  (A) tflite 로딩 / 전처리 문제   인지
  (B) yolo12n 박스에서 만든 크롭이 학습 크롭과 달라서   인지
  를 가른다.

방법: ppe_eval_compare.py 의 MBCModel.predict 와 *완전히 동일한* 전처리로,
      train_sampled 의 이미지(= 모델이 학습 때 본 형태)를 그대로 넣어서
      파일명 라벨(XXXX_HV)과 비교한다.

해석:
  - 노트북의 정확도(~80%대)와 비슷하게 나옴  -> tflite/전처리 정상.
                                              크롭 파이프라인이 원인 (B).
  - 노트북보다 한참 낮게 나옴               -> tflite 로딩/전처리가 원인 (A).
================================================================================
"""

import os
import glob

import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise SystemExit("tensorflow 가 필요합니다:  pip install tensorflow")


# ==============================================================================
# ★ 설정
# ==============================================================================
MBC_MODEL_PATH = r"D:\AIProject\YSY\PPE_MobileNetV3Large.tflite"   # float tflite
# train_sampled 폴더 (XXXX_HV.jpg 형식 파일들이 들어있는 곳) — 본인 경로로 수정
TRAIN_SAMPLED_DIR = r"D:\AIProject\YSY\train_sampled"

MBC_INPUT_SIZE = 224
MBC_THRESHOLD  = 0.5            # ppe_eval_compare.py 와 동일. 0.6 도 따로 시험해볼 것
MAX_IMAGES     = 800            # 너무 많으면 일부만
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


# ==============================================================================
# 전처리 / 모델 — ppe_eval_compare.py 의 MBCModel 과 동일 로직
# ==============================================================================
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


def load_interpreter(model_path):
    OpResolver = tf.lite.experimental.OpResolverType
    last_err = None
    for resolver in (OpResolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                     OpResolver.BUILTIN_REF):
        try:
            interp = tf.lite.Interpreter(model_path=model_path,
                                         experimental_op_resolver_type=resolver)
            interp.allocate_tensors()
            print(f"interpreter 로드 OK (resolver={resolver.name})")
            return interp
        except (RuntimeError, ValueError) as e:
            last_err = e
    raise SystemExit(f"모델 로드 실패: {last_err}")


class MBCModel:
    def __init__(self, model_path):
        self.interp = load_interpreter(model_path)
        self.in_det = self.interp.get_input_details()
        self.out_det = self.interp.get_output_details()
        self.in_dtype = self.in_det[0]["dtype"]
        self.out_dtype = self.out_det[0]["dtype"]
        self.in_scale, self.in_zp = self.in_det[0]["quantization"]
        self.out_scale, self.out_zp = self.out_det[0]["quantization"]
        print(f"입력 dtype={self.in_dtype.__name__}, "
              f"양자화(scale={self.in_scale}, zp={self.in_zp})")

    def predict(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inp = resize_with_padding(rgb, MBC_INPUT_SIZE).astype(np.float32)
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
        if self.out_dtype in (np.int8, np.uint8) and self.out_scale not in (0, None):
            out = self.out_scale * (out - self.out_zp)
        return float(out[0]), float(out[1])


def parse_label_from_filename(filepath):
    """XXXX_HV.jpg -> (helmet, vest).  실패 시 None."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    if "_" not in name:
        return None
    tag = name.split("_")[-1]
    if len(tag) < 2 or not tag[0].isdigit() or not tag[1].isdigit():
        return None
    h, v = int(tag[0]), int(tag[1])
    if h not in (0, 1) or v not in (0, 1):
        return None
    return h, v


# ==============================================================================
# 메인
# ==============================================================================
def main():
    if not os.path.isdir(TRAIN_SAMPLED_DIR):
        raise SystemExit(f"train_sampled 폴더를 찾을 수 없음: {TRAIN_SAMPLED_DIR}\n"
                         f"  -> TRAIN_SAMPLED_DIR 경로를 수정하세요.")

    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(TRAIN_SAMPLED_DIR, ext)))
    files = sorted(files)[:MAX_IMAGES]
    print(f"train_sampled 이미지: {len(files)}장 평가\n")

    model = MBCModel(MBC_MODEL_PATH)

    # 혼동행렬 카운트  [gt][pred]
    h_cm = [[0, 0], [0, 0]]
    v_cm = [[0, 0], [0, 0]]
    skipped = 0

    for i, fp in enumerate(files, start=1):
        lbl = parse_label_from_filename(fp)
        if lbl is None:
            skipped += 1
            continue
        gt_h, gt_v = lbl
        img = cv2.imread(fp)
        if img is None:
            skipped += 1
            continue
        h_prob, v_prob = model.predict(img)
        pred_h = int(h_prob >= MBC_THRESHOLD)
        pred_v = int(v_prob >= MBC_THRESHOLD)
        h_cm[gt_h][pred_h] += 1
        v_cm[gt_v][pred_v] += 1
        if i % 100 == 0:
            print(f"  ...{i}/{len(files)}")

    def report(name, cm):
        total = sum(cm[0]) + sum(cm[1])
        if total == 0:
            print(f"\n[{name}] 평가 데이터 없음")
            return
        correct = cm[0][0] + cm[1][1]
        acc = correct / total
        # FNR = 실제 미착용(0) 인데 착용(1) 으로 예측
        neg = sum(cm[0])
        fnr = (cm[0][1] / neg) if neg else None
        print(f"\n[{name}]  (n={total})")
        print(f"  정확도: {acc * 100:.1f}%")
        print(f"  FNR   : {'N/A' if fnr is None else f'{fnr * 100:.1f}%'}"
              f"   (실제 미착용 {neg}건 중 착용 오판 {cm[0][1]}건)")
        print(f"  혼동행렬  [GT행 / 예측열]")
        print(f"            예측0   예측1")
        print(f"    GT 0    {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"    GT 1    {cm[1][0]:5d}  {cm[1][1]:5d}")

    print("\n" + "=" * 50)
    print(f"MBC 직접 테스트 결과  (threshold={MBC_THRESHOLD})")
    print("=" * 50)
    report("HELMET", h_cm)
    report("VEST", v_cm)
    print(f"\n건너뜀(라벨 파싱 실패 등): {skipped}장")
    print("\n해석:")
    print("  노트북 정확도와 비슷 -> tflite/전처리 정상, 크롭 파이프라인이 원인")
    print("  노트북보다 한참 낮음 -> tflite 로딩/전처리가 원인")


if __name__ == "__main__":
    main()

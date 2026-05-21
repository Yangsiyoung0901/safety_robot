"""
detector.py — PPE 분류기 + 유틸

제공:
  - PPEClassifierTFLite  (.tflite 모델용, tflite_runtime 필요)
  - PPEClassifier        (.pth 모델용, torch 필요) — torch 미설치 시 정의되지 않음
  - create_ppe_classifier(path, threshold) — 확장자 보고 자동 선택
  - compute_upper_body_box, _is_head_only — 상체 크롭 유틸
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# PyTorch (선택)
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    torch = None

# TFLite (선택)
TFLITE_AVAILABLE = False
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        pass


# =============================================================
# 상체 크롭 함수
# =============================================================

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def compute_upper_body_box(
    x1: int, y1: int, x2: int, y2: int,
    img_w: int, img_h: int,
    expand: float = 0.08,
    upper_ratio: float = 0.82,
) -> tuple | None:
    """person bbox에서 상체 영역을 계산한다."""
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
    crop_box: tuple,
    min_crop_height: int = 90,
    min_crop_aspect: float = 0.90,
) -> bool:
    """크롭이 머리만 잡힌 것인지 판단"""
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
# PPE Classification — TFLite 추론기
# =============================================================

class PPEClassifierTFLite:
    """
    TFLite 모델로 PPE 착용 여부 분류.
    INT8/FP32 TFLite 모두 지원.
    """

    def __init__(self, model_path: str, threshold: float = 0.3, output_is_logits: bool = True):
        if not TFLITE_AVAILABLE:
            raise RuntimeError("tflite_runtime 또는 tensorflow가 필요합니다")

        self.threshold = threshold
        self.output_is_logits = output_is_logits
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]['shape']
        self.input_h = input_shape[1]
        self.input_w = input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']

        self.input_scale = 1.0
        self.input_zero_point = 0
        if self.input_dtype in (np.uint8, np.int8):
            quant = self.input_details[0].get('quantization_parameters', {})
            scales = quant.get('scales', [1.0])
            zero_points = quant.get('zero_points', [0])
            if len(scales) > 0:
                self.input_scale = scales[0]
            if len(zero_points) > 0:
                self.input_zero_point = zero_points[0]

        print(f"[PPEClassifierTFLite] 입력: {input_shape}, dtype: {self.input_dtype}")

    def classify(self, crop_bgr: np.ndarray) -> tuple:
        """크롭 이미지 → (helmet_worn, vest_worn, helmet_prob, vest_prob)"""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (self.input_w, self.input_h))

        if self.input_dtype == np.float32:
            img = resized.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            input_data = np.expand_dims(img.astype(np.float32), axis=0)
        elif self.input_dtype in (np.uint8, np.int8):
            # Quantized model input: normalize then quantize using input_scale/zero_point
            img = resized.astype(np.float32) / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            # Protect against zero scale
            scale = float(self.input_scale) if self.input_scale != 0 else 1.0
            zp = int(self.input_zero_point)
            if self.input_dtype == np.uint8:
                q = np.clip(np.round(img / scale + zp), 0, 255).astype(np.uint8)
            else:
                q = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)
            input_data = np.expand_dims(q, axis=0)
        else:
            input_data = np.expand_dims(resized.astype(np.float32), axis=0)
        else:
            input_data = np.expand_dims(resized.astype(np.float32), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # 역양자화 (INT8/UINT8 출력인 경우)
        out_detail = self.output_details[0]
        if out_detail['dtype'] != np.float32:
            out_quant = out_detail.get('quantization_parameters', {})
            out_scale = out_quant.get('scales', [1.0])
            out_zp = out_quant.get('zero_points', [0])
            if len(out_scale) > 0 and out_scale[0] != 0:
                output = (output.astype(np.float32) - out_zp[0]) * out_scale[0]

        values = output.flatten().astype(np.float64)

        # 확률/로그잇 자동 감지 로직
        # 1) 명시적으로 output_is_logits=True이면 시그모이드 적용
        # 2) output_is_logits=False인 경우라도 출력값이 [0,1] 범위를 벗어나면 logits로 간주하여 시그모이드 적용
        # 3) 출력값이 [0,1] 범위에 있고 합이 거의 1.0이면(softmax 가능성) 그대로 사용하되 로그로 알림
        if self.output_is_logits:
            probs = 1.0 / (1.0 + np.exp(-values))
        else:
            # 값 범위 검사
            if np.any(values < 0.0) or np.any(values > 1.0):
                # 의심되는 logits — 시그모이드 적용
                probs = 1.0 / (1.0 + np.exp(-values))
                print("[PPEClassifierTFLite] Warning: outputs appear to be logits (outside [0,1]). Applied sigmoid to convert to probabilities.")
            elif len(values) == 2 and abs(np.sum(values) - 1.0) < 1e-3 and np.all(values >= 0.0) and np.all(values <= 1.0):
                # softmax로 보이는 출력 — 두 값이 합해서 1인 경우
                probs = values
                print("[PPEClassifierTFLite] Notice: outputs sum to 1.0 (possible softmax). Using values as probabilities.")
            else:
                # 이미 확률로 보이는 경우
                probs = values

        h_prob = float(probs[0]) if len(probs) > 0 else 0.0
        v_prob = float(probs[1]) if len(probs) > 1 else 0.0

        # 상세 디버깅: 확률이 0.5 근처이거나 환경변수 MBC_DEBUG=1이면 원인 추적용 로그 출력
        try:
            dbg_env = os.environ.get("MBC_DEBUG", "0") == "1"
        except Exception:
            dbg_env = False

        if dbg_env or (0.45 < h_prob < 0.55) or (0.45 < v_prob < 0.55):
            try:
                inp = resized.astype(np.float32) / 255.0
                inp_mean = float(np.mean(inp))
                inp_std = float(np.std(inp))
            except Exception:
                inp_mean = None
                inp_std = None

            out_dtype = out_detail.get('dtype', 'unknown') if 'out_detail' in locals() else 'unknown'
            print(
                f"[PPEClassifierTFLite DEBUG] input_mean={inp_mean:.4f} input_std={inp_std:.4f} \\n+  out_dtype={out_dtype} output_raw={values.tolist()} probs={probs.tolist()} \\n+  h_prob={h_prob:.4f} v_prob={v_prob:.4f} threshold={self.threshold} output_is_logits={self.output_is_logits}"
            )

        return h_prob >= self.threshold, v_prob >= self.threshold, h_prob, v_prob


# =============================================================
# PPE Classification — PyTorch 추론기
# torch 미설치 시 이 클래스는 정의되지 않는다.
# =============================================================

if TORCH_AVAILABLE:
    class PPEClassifier:
        """
        PyTorch MobileNetV3Large 기반 PPE 착용 여부 분류.
        .pth 파일 전용.
        """

        def __init__(self, model_path: str, device, threshold: float = 0.3):
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

        def _load_model(self, model_path: str):
            model = models.mobilenet_v3_large(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 2)

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model

        def classify(self, crop_bgr: np.ndarray) -> tuple:
            """크롭 이미지 → (helmet_worn, vest_worn, helmet_prob, vest_prob)"""
            with torch.no_grad():
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu()

                h_prob = probs[0].item()
                v_prob = probs[1].item()
                return h_prob >= self.threshold, v_prob >= self.threshold, h_prob, v_prob


# =============================================================
# 팩토리 함수
# =============================================================

def create_ppe_classifier(model_path: str, threshold: float = 0.3, output_is_logits: bool = True):
    """
    모델 파일 확장자를 보고 TFLite 또는 PyTorch 분류기를 생성한다.
    .tflite → PPEClassifierTFLite
    .pth    → PPEClassifier (PyTorch)

    output_is_logits: TFLite 모델 출력이 logits이면 True (sigmoid 적용),
                      이미 확률이면 False (sigmoid 건너뜀).
                      PyTorch 모델은 항상 logits 출력이므로 이 옵션 무시.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

    ext = path.suffix.lower()

    if ext == ".tflite":
        if not TFLITE_AVAILABLE:
            raise RuntimeError(f"TFLite 런타임 미설치 — {model_path} 로드 불가")
        return PPEClassifierTFLite(str(path), threshold, output_is_logits=output_is_logits)

    elif ext in (".pth", ".pt"):
        if not TORCH_AVAILABLE:
            raise RuntimeError(f"PyTorch 미설치 — {model_path} 로드 불가")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return PPEClassifier(str(path), device, threshold)

    else:
        raise ValueError(f"지원하지 않는 모델 형식: {ext} ({model_path})")
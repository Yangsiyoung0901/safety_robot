# Safe Eye 코드 수정 로그

## 1. TFLite / PyTorch PPE 분류기 로딩 개선

- `detector.py`에 TFLite용 `PPEClassifierTFLite` 추가
- `.tflite`, `.pth`, `.pt` 확장자에 따라 분류기를 자동 생성하는 `create_ppe_classifier()` 추가
- `safe_eye_danger.py`에서 기존 `PPEClassifier` 직접 생성 대신 `create_ppe_classifier()` 사용
- `torch`가 없는 환경에서도 TFLite 모델 import가 깨지지 않도록 PyTorch 분류기 정의를 조건부 처리

## 2. PPE 판정 실패 처리 개선

- MBC 분류가 실패하는 경우 OD 모델로 fallback하도록 변경
- PPE 모델이 아예 없는 경우 `unavailable` 상태로 처리
- PPE 모델 미탑재 상태에서는 헬멧/조끼 미착용 음성 경고를 발생시키지 않도록 수정
- 위험 구역 진입 경고는 PPE unavailable 상태에서도 유지

## 3. Camera-only 조건 수정

- 기존에는 PPE 모델이 없으면 person/danger 감지도 함께 비활성화됨
- 수정 후 `person_model`, `ppe_model`, `mbc_classifier`, `danger_sign_model_path`가 모두 없을 때만 camera-only 모드 진입
- PPE 모델만 없는 경우에도 사람 감지와 위험 표지 감지는 계속 동작

## 4. TFLite 출력 처리 수정

- TFLite 출력에 sigmoid가 이중 적용될 수 있는 문제 수정
- 자동 판별 방식 제거
- `output_is_logits` 명시 파라미터 추가
- 기본값은 `True`
- 확률 출력 모델을 사용할 경우 `create_ppe_classifier(path, threshold, output_is_logits=False)`로 설정 가능

## 5. INT8 양자화 안정화

- INT8 TFLite 입력 변환 시 값이 `[-128, 127]` 범위를 벗어나 wraparound 될 수 있는 문제 수정
- `np.clip(..., -128, 127).astype(np.int8)` 적용

## 6. 위험 표지 캐시 버그 수정

- `danger_detector.py`에서 재탐지 결과가 빈 리스트일 때도 캐시를 갱신하도록 변경
- 기존에는 한 번 표지판이 감지되면 이후 사라져도 이전 캐시가 유지될 수 있었음
- 수정 후 표지판이 사라지면 위험 구역도 정상 해제됨

## 7. PPE 표시 라벨 개선

- 기존에는 전역 `use_mbc` / `use_od` 기준으로 라벨 표시
- 수정 후 사람별 `person_method` 기준으로 표시
- 표시값:
  - `MBC`: MBC 분류 사용
  - `OD`: OD 모델 사용
  - `OD*`: MBC 실패 후 OD fallback
  - `PPE N/A`: PPE 판정 불가

## 8. 테스트 스크립트 수정

- `run_detection_test.py`의 모델 경로 탐색 오류 수정
- 기존 `parents[2] / "models"`에서 `parents[1] / "models"`로 변경
- 사라진 `Detector` 클래스 의존 제거
- `create_ppe_classifier()`와 YOLO person detector 기반 테스트 흐름으로 수정

## 검증 결과

- ZIP 내 Python 파일 syntax compile 통과
- 최종 포함 파일:
  - `safe_eye_danger.py`
  - `detector.py`
  - `danger_detector.py`
- 남은 검증 필요 사항:
  - Raspberry Pi 환경에서 카메라, GPIO, 스피커 동작 확인
  - 실제 TFLite / YOLO 모델 추론 결과 확인
  - `output_is_logits` 값이 현재 TFLite 모델 출력 형식과 맞는지 확인
```
# vision 모듈 설명서

# 다른 코드에서는 from vision.detector import Detector, PersonDetection
# 하면 vision 폴더 안에 있는 Detector 클래스와 PersonDetection 데이터를
# 불러와서 쓸 수 있습니다. __init__.py는 패키지 인식용이라 삭제하면 import 안 됩니다.

# =============================================================
# 폴더 구조
# =============================================================
#
# safety_robot/
# ├── main.py                  ← 진입점. 카메라 + 감지 + 화면 출력 루프 (PC 테스트용)
# ├── config.yaml              ← 모든 설정값 (임계값, 모델 경로, 크롭 파라미터)
# └── vision/
#     ├── __init__.py           ← 패키지 인식용 (비어있음)
#     ├── camera.py             ← 웹캠 캡처 (Camera 클래스)
#     └── detector.py           ← Person / PPE / Danger 통합 파이프라인

# =============================================================
# detector.py 핵심 — 하이브리드 PPE 판정
# =============================================================
#
# detector.py의 원래 용도는 Person / PPE / Danger 3가지를 통합 처리하는 파이프라인이다.
# 이 중 PPE 판정 부분을 하이브리드 방식으로 구현했다.
#
# 하이브리드란:
#   - 사람이 1명이면 → Classification 방식 (내 담당)
#   - 사람이 2명 이상이면 → OD 방식 (팀원 코드에서 처리)
#
# 이렇게 나눈 이유:
#   - 1명일 때: Classification이 OD보다 추론 속도가 빠르고, 크롭이 깨끗해서 정확도도 높음
#   - 2명 이상일 때: Classification은 크롭에 옆 사람이 섞여서 정확도가 떨어짐 (미경고율 32%)
#                    OD는 다인원에서도 헬멧 정확도 91%로 안정적
#   - 발표 자료 Section 4 참고: OD vs CNN-MBC 정량 비교 결과에 근거한 선택

# =============================================================
# 매커니즘
# =============================================================
#
# main.py에서 카메라, Detector를 초기화
# config.yaml에서 모든 설정값을 읽어옴
#
# -> 카메라가 웹캠에서 프레임을 1장씩 읽음
# -> 프레임을 Detector.detect()에 전달
#
#    Detector 내부 처리 순서 (detector.py):
#    ┌─────────────────────────────────────────────────────────┐
#    │ 1단계: YOLO Person Detection (공통)                     │
#    │   - YOLOv5n 모델로 프레임에서 사람(class 0)을 감지     │
#    │   - 신뢰도 0.5 이상인 사람만 남김                       │
#    │   - 너무 작은 bbox는 제외                               │
#    │   - 왼쪽→오른쪽으로 정렬 (x1 기준)                     │
#    │                                                         │
#    │ 2단계: 하이브리드 PPE 분기                              │
#    │                                                         │
#    │   ┌─ 사람 1명일 때 ──────────────────────────────────┐  │
#    │   │ - person bbox에서 상위 82%만 잘라서 상체 크롭     │  │
#    │   │ - 좌우 8%, 상단/하단에 패딩 추가                   │  │
#    │   │ - 머리만 잡힌 크롭은 제외 (높이 90px 미만)        │  │
#    │   │ - 크롭 이미지를 224×224로 리사이즈                 │  │
#    │   │ - MobileNetV3Large 모델에 입력                     │  │
#    │   │ - sigmoid 출력: [helmet_prob, vest_prob]            │  │
#    │   │ - 확률 ≥ 0.3이면 "착용", 미만이면 "미착용"        │  │
#    │   │ - method = "classification"                         │  │
#    │   └────────────────────────────────────────────────────┘  │
#    │                                                         │
#    │   ┌─ 사람 2명 이상일 때 ─────────────────────────────┐  │
#    │   │ - helmet/vest를 None으로 남겨둠                    │  │
#    │   │ - 팀원 코드(safe_eye_monitor.py)의 PPE OD 모델이  │  │
#    │   │   helmet·vest bbox를 직접 검출해서 채움            │  │
#    │   │ - method = "" (OD에서 처리 후 "od"로 변경)         │  │
#    │   └────────────────────────────────────────────────────┘  │
#    │                                                         │
#    │ 3단계: Danger Detection (팀원 코드에서 처리)            │
#    │   - 위험 표지 YOLO 모델은 팀원 코드에서 로드·실행      │
#    │   - 결과는 PersonDetection.in_danger에 반영             │
#    └─────────────────────────────────────────────────────────┘
#
# -> Detector가 PersonDetection 리스트를 반환
#    PersonDetection에 담긴 정보:
#      - bbox: 사람 위치 (x1, y1, x2, y2)
#      - confidence: 사람 감지 신뢰도
#      - crop_box: 상체 크롭 영역 (classification일 때만)
#      - helmet: 헬멧 착용 여부 (True/False/None)
#      - vest: 조끼 착용 여부 (True/False/None)
#      - helmet_prob, vest_prob: 착용 확률값
#      - in_danger: 위험 구역 근접 여부
#      - method: "classification" / "od" / ""
#
# -> main.py에서 결과를 화면에 오버레이
#    - person bbox + 번호 (왼쪽부터 #1, #2, ...)
#    - 상체 크롭 영역 (classification일 때만 표시)
#    - 착용=녹색, 미착용=주황
#    - 현재 PPE 방식 표시 ("Classification (1명)" / "OD (2명+)")
#    - FPS, 인원 수, 위반 수 표시

# =============================================================
# 팀원 코드(safe_eye_monitor.py)와의 연동
# =============================================================
#
# 팀원 코드가 메인 시스템이고, 이 코드는 거기에 끼워넣는 모듈이다.
#
# 연동 방법:
#   1. detector.py의 Detector를 팀원 코드에서 import
#   2. detect() 호출 → PersonDetection 리스트 받음
#   3. det.helmet이 None이 아니면 → Classification 결과 그대로 사용
#   4. det.helmet이 None이면 → 팀원 코드의 OD 결과로 채움
#
# 예시:
#   detections = detector.detect(frame)
#   for det in detections:
#       if det.helmet is None:
#           # OD 방식으로 판정 (팀원 코드)
#           det.helmet = od_result_helmet
#           det.vest = od_result_vest
#           det.method = "od"

# =============================================================
# 실행 방법
# =============================================================
#
# 설치:
#   pip install ultralytics opencv-python torch torchvision pyyaml
#
# 실행 (PC 테스트):
#   cd safety_robot
#   python main.py
#
# PPE Classification 모델 연결:
#   config.yaml에서:
#     models:
#       ppe_classifier: "models/ppe_mobilenetv3.pth"

# =============================================================
# config.yaml 주요 설정값
# =============================================================
#
# models.person_detector      ← YOLO Person 모델
# models.ppe_classifier       ← MobileNetV3 분류 모델 (1명일 때 사용)
# models.ppe_od_model         ← PPE OD 모델 (2명+ 팀원 코드에서 사용)
# models.danger_detector      ← 위험 표지 모델 (팀원 코드에서 사용)
#
# thresholds.person_confidence: 0.5
# thresholds.ppe_threshold: 0.3      ← Classification sigmoid 임계값
# thresholds.ppe_od_confidence: 0.35 ← OD 신뢰도 임계값
#
# crop.upper_ratio: 0.82     ← person 높이의 82%를 상체로 간주
# crop.expand: 0.08          ← bbox 주변 8% 패딩
# crop.min_crop_height: 90   ← 이보다 작으면 머리만 잡힌 것으로 제외

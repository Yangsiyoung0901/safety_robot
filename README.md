# Safe Eye — 산업 현장 안전 감지 시스템

## 실행 방법

```bash
cd safe_eye
python3 ui/safe_eye_danger.py --host 0.0.0.0 --port 8000
```

브라우저에서 `http://<라즈베리파이IP>:8000` 접속

## 폴더 구조

```
safe_eye/
├── ui/                          ← 웹 서버 (진입점)
│   ├── safe_eye_danger.py       ← 통합 웹 모니터 서버 (여기서 실행)
│   └── index.html               ← 브라우저 대시보드
│
├── models/                      ← 모델 파일 (여기에 배치)
│   ├── yolo11n.pt               ← Person Detector (COCO 사전학습)
│   ├── best_p.pt                ← PPE OD 모델 (helmet/vest bbox)
│   ├── PPE_MobileNetV3Large_INT8.tflite  ← PPE 분류 모델 (MBC, 1명용)
│   └── signs_b.pt              ← 위험 표지 탐지 모델
│
├── Main/                        ← AI 파이프라인 모듈
│   └── detector.py              ← PPEClassifier (TFLite/PyTorch 자동 선택)
│
├── Danger/                      ← 위험 표지판 감지 모듈
│   ├── __init__.py
│   └── danger_detector.py       ← DangerDetector
│
├── vision/                      ← 카메라 모듈
│   ├── __init__.py
│   └── camera.py                ← LatestFrameCamera
│
├── sensor/                      ← IR 센서 모듈
│   ├── __init__.py
│   └── ir_sensor.py             ← IRSensor
│
├── speaker/                     ← 음성 경고 모듈
│   ├── __init__.py
│   ├── speaker.py               ← DangerSpeaker
│   └── assets/                  ← WAV 파일
│       ├── danger_zone_ko.wav
│       ├── no_helmet_ko.wav
│       └── no_vest_ko.wav
│
├── tools/                       ← 테스트 도구
│   └── run_detection_test.py    ← 감지 테스트 스크립트
│
└── docs/                        ← 문서
```

## 동작 흐름

```
IR 센서 감지 → 카메라 최신 프레임 → YOLO person 감지
  → 1명: MBC 분류 (MobileNetV3, 상체 크롭)
  → 2명+: YOLO OD (best_p.pt, helmet/vest bbox)
  → 위험 표지판 감지 (signs_b.pt) → 위험 구역 판정
  → 위반 시 스피커 경고 + 화면 표시
  → 브라우저 대시보드 실시간 업데이트
```

## 하이브리드 PPE 판정

| 인원 | 방식 | 모델 |
|------|------|------|
| 1명 | MBC 분류 | PPE_MobileNetV3Large_INT8.tflite |
| 2명+ | YOLO OD | best_p.pt |
| MBC 실패 시 | OD fallback | best_p.pt |
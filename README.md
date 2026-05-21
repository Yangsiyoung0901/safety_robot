# Safe Eye — On-Device AI 기반 산업 현장 안전 감지 시스템

> 고정 카메라 비전으로 작업자의 PPE(안전모·안전조끼) 미착용과 위험 구역 근접을
> **네트워크 없이 단말에서 실시간 감지**하고 즉시 음성·시각 경고를 출력하는 시스템.
>
> Raspberry Pi 5 + Camera Module · YOLOv11 · MobileNetV3Large · TFLite · 2026
> AI 로봇 SW 개발자 교육과정 · On-Device AI 프로젝트

---

## 핵심 성과

| 지표 | 목표 | 달성 |
|------|------|------|
| 헬멧 감지 정확도 | ≥ 80% | **OD 91.6% · MLC 83.7%** ✓ |
| 조끼 감지 정확도 | ≥ 70% | **OD 81.8% · MLC 81.8%** ✓ |
| 실시간 추론 속도 | ≥ 5 FPS | **Pi5 기준 6 FPS 이상** ✓ |
| 시스템 통합 동작 | IR → 감지 → 판별 → 경고 | **단일 흐름 완성** ✓ |

- 단일 모델 한계를 보완하는 **OD/MLC 하이브리드 PPE 판별 파이프라인** 설계
- OD 학습셋과 분리된 **11클래스 독립 GT 데이터셋**으로 누수 없는 공정 비교 (총 1,477명 평가)
- IR 트리거 기반 **저전력 추론 활성화** 구조 — 상시 추론 대비 자원 부담 감소

---

## 팀 구성

| 멤버 | 역할 |
|------|------|
| **양시영** | 시스템 및 아키텍쳐 설계, PPE 모델 정량적 성능 평가 |
| **류상균** | PPE OD 객체 분류 모델 설계 및 학습, 하드웨어 및 임베디드 구동 시스템 구현 |
| **채윤식** | PPE MLC 모델 설계 및 학습, 위험 지역 감지 시스템 |
| **박희연** | 학습 데이터 및 성능 평가 데이터 전처리, 시스템 통합 및 프론트엔드 위젯 |

---

## 목차

- [무엇을 만들었나](#무엇을-만들었나)
- [동작 흐름](#동작-흐름)
- [하이브리드 PPE 판별 파이프라인](#하이브리드-ppe-판별-파이프라인)
- [위험 구역 감지](#위험-구역-감지)
- [하드웨어 구성](#하드웨어-구성)
- [소프트웨어 아키텍처](#소프트웨어-아키텍처)
- [디렉터리 구조](#디렉터리-구조)
- [모델 구성](#모델-구성)
- [데이터셋](#데이터셋)
- [평가 방법론](#평가-방법론)
- [설치 및 실행](#설치-및-실행)
- [데모 시나리오](#데모-시나리오)
- [주요 결정과 학습](#주요-결정과-학습)
- [한계 및 향후 과제](#한계-및-향후-과제)
- [변경 이력](#변경-이력)
- [참고 자료](#참고-자료)

---

## 무엇을 만들었나

산업재해 사망자는 매년 800명대 — 추락·끼임이 다수다. 사람 감독만으로는 24시간·전 구역의 사각지대를 메울 수 없고, 네트워크가 불안정한 현장일수록 위험은 커진다.

**Safe Eye**는 단말에서 추론이 끝나는 On-Device AI 안전 비전 시스템이다.

- **IR 센서 트리거** → **Person Detector(YOLO)** → **PPE 판별(OD/MLC)** + **위험구역 판정(OD)**을 단말에서 동시 수행
- 결과를 **카메라 오버레이 화면**과 **음성**으로 현장에 즉시 전달
- PPE 검출 정확도를 높이기 위해 **OD·MLC 두 접근법을 정량 비교**하고, 환경별 최적 조합으로 **하이브리드 적용**

| 특징 | 설명 |
|------|------|
| **올인원** | PPE 판별 + 위험구역 접근 판정 + 오버레이·음성 경고 — 단말 하나로 완결 |
| **On-Device** | 네트워크 없이 단말 추론 — 통신 단절 환경에서도 안전 판단 지속 |
| **하이브리드** | 1인 환경 MLC(정확도) + 다인원 OD(실시간성) 자동 분기 |

---

## 동작 흐름

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ ① IR     │ → │ ② 사람   │ → │ ③ PPE    │ → │ ④ 위험   │ → │ ⑤ 현장   │
│  트리거  │   │  감지    │   │  판별    │   │  구역    │   │  경고    │
│          │   │  (YOLO)  │   │ (OD/MLC) │   │  판정    │   │ (오버레이│
│ 사람 접근│   │ person   │   │ 환경별   │   │ 좌표 ×   │   │ + 음성)  │
│ → AI 활성│   │ bbox     │   │ 자동 선택│   │ 영역 교차│   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                    ↑               ↑
                                    └──── 동시 수행 ─┘
```

③ PPE 판별과 ④ 위험구역 판정은 동시 수행 — 단말 하나에서 두 안전 항목을 병렬로 평가.

| 단계 | 효율화 포인트 |
|------|--------------|
| ① IR 트리거 | 상시 추론 대신 사람 접근 시에만 AI 활성화 — 전력·CPU 부담 감소 |
| ② 사람 감지 | OD·MLC·Danger Zone가 동일한 person bbox 공유 — detector 편향 제거 |
| ③ PPE 판별 | 인원수 기반 모델 자동 분기 (1명 → MLC, 2명+ → OD) |
| ④ 위험구역 | 30프레임 단위 표지판 탐지 후 캐시 재사용 — 자원 부담 최소화 |
| ⑤ 현장 경고 | 30초 쿨다운, 위험구역 > PPE 우선순위 |

---

## 하이브리드 PPE 판별 파이프라인

본 프로젝트의 핵심 기술적 기여는 **OD 방식과 MLC 방식의 정량 비교**를 거쳐 **환경별 최적 모델을 자동 선택하는 하이브리드 파이프라인**이다.

### 두 가지 접근법

**OD (Object Detection) 방식 — 객체 탐지**
- YOLO OD가 helmet·vest bbox를 직접 검출 → 각 person bbox에 귀속 → 착용 여부 판정
- 이미지당 추론 1회 (인원수 무관) → 다인원 환경에서도 일정한 FPS 유지
- vest 검출 정확도가 학습된 OD 모델 품질에 직접 의존

**MLC (Multi-Label Classification) 방식 — 분류**
- person bbox 상체 크롭 → MobileNetV3Large 멀티 라벨 분류 → helmet·vest 각각 착용 확률
- 사람 단위 크롭을 분류 — 사람 수만큼 추론 반복
- 1인 환경에서 정확도 우위 (특히 vest), 다인원에서 FPS 선형 하락 + 크롭 오염 가능성

### 환경별 자동 선택 근거

| 환경 | 평균 인원 | Vest 정확도 우위 | FPS (PC 기준) | **채택** |
|------|----------|-----------------|--------------|----------|
| Single (1인) | 1.19명 | **MLC 90.8% vs OD 79.6% (+11.2%p)** | MLC 3.48 / OD 3.70 (차이 미미) | **MLC** |
| Crowd (3인+) | 5.39명 | MLC 77.8% vs OD 73.6% (+4.2%p, 작음) | **MLC 1.92 / OD 3.70 (MLC 절반)** | **OD** |

- **1인 환경**: vest 11.2%p 차이로 MLC 압도, FPS 차이는 미미 → 정확도 기준으로 MLC 채택
- **다인원 환경**: MLC FPS 절반으로 무너짐(3.5→1.9), 정확도 차이 작음 → 실시간성 확보로 OD 채택, vest 4.2%p 손실 감수

### 전체 정확도 (Overall / Single / Crowd)

| 환경 | Helmet OD | Helmet MLC | Vest OD | Vest MLC |
|------|-----------|------------|---------|----------|
| Single (1인, n=588) | 92.6% | 91.6% | 79.6% | **90.8%** |
| Crowd (3인+, n=889) | **92.8%** | 89.6% | **73.6%** | 77.8% |
| **Overall** | **92.8%** | 90.4% | 76.0% | **83.0%** |

> **헬멧** → OD 일관 우세 / **조끼** → MLC 일관 우세. 단일 모델로는 정확도·실시간성 동시 만족 불가.

### 핵심 발견

1. **클래스별 우세 모델이 다름** — 헬멧 OD, 조끼 MLC
2. **다인원에서 양 모델 vest 동반 하락** — OD 79.6→73.6%, MLC 90.8→77.8%. 가림·작은 bbox 영향
3. **MLC는 사람 수에 따라 FPS 선형 하락** — single 3.48 → crowd 1.92 FPS. OD는 인원 무관 3.70 FPS 일정
4. **MLC만 임계값 조정 가능** — sigmoid score sweep으로 PR Curve 운영점 조절. OD는 단일 운영 포인트

---

## 위험 구역 감지

표지판 기반 자동 영역 설정 방식. 위험 표지판을 한 번 탐지하면 그 bbox를 확장해 위험 구역 polygon을 만들고, 작업자 발 위치가 영역 안에 들어오면 즉시 DANGER 경고.

### 동작 구조

```
① YOLOv11n으로 위험 표지판 탐지
   고정 위치 표지판 → 30프레임마다 탐지 후 결과 캐시 재사용
                       │
② 표지판 bbox 기준 Danger Zone 설정
   bbox를 확장해 위험 구역 polygon 생성 (설치 환경별 튜닝)
                       │
③ 발 위치 기반 진입 판정 · 즉시 경고
   person bbox 하단 중심점이 polygon 내 → DANGER 경고 발령
```

### 학습 결과 (YOLO11n · 1,915장 · img 416)

| 지표 | 값 |
|------|---|
| Precision | **1.000** |
| Recall | 0.940 |
| F1 | 0.810 |
| mAP@0.5 | 0.787 |

> 표지판이 정확한 위치에 있어야 위험 구역이 올바로 설정되므로 **Precision 100%**를 우선 최적화.

---

## 하드웨어 구성

```
┌──────────────────────────────────────────────────┐
│           Safe Eye 안전 감지 모듈                 │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │           Raspberry Pi 5                 │    │
│  │      (메인 컨트롤 / AI 추론)             │    │
│  └──┬──────────┬─────────────┬─────────────┘    │
│     │          │             │                   │
│  ┌──▼──────┐ ┌─▼────────┐ ┌─▼─────────┐         │
│  │ Camera  │ │ IR 센서  │ │ USB       │         │
│  │ Module  │ │ (트리거) │ │ 스피커    │         │
│  │ (CSI)   │ │          │ │ (음성경고)│         │
│  └─────────┘ └──────────┘ └───────────┘         │
└──────────────────────────────────────────────────┘
```

| 구성품 | 사양 / 역할 |
|--------|-------------|
| **Raspberry Pi 5** | Cortex-A76 × 4코어 @ 2.4GHz · 4GB+ · Raspberry Pi OS 64-bit (Bookworm) |
| **Camera Module 3** | 640×480 / 30fps · MIPI CSI-2 · 고정 마운트 |
| **IR 센서** | 사람 접근 트리거 → AI 추론 활성화 신호 |
| **USB 스피커** | pygame 기반 음성 경고 출력 |

---

## 소프트웨어 아키텍처

### 모듈 구성

| 모듈 | 클래스 | 역할 |
|------|--------|------|
| `sensor/ir_sensor.py` | `IRSensor` | 사람 감지 트리거 — 미감지 시 YOLO 추론 중단 |
| `vision/camera.py` | `LatestFrameCamera` | 최신 프레임 1장만 버퍼링 — 콜드 스타트 없음 |
| `Main/detector.py` | `PPEClassifier` | OD/MLC 하이브리드 PPE 판별 (TFLite/PyTorch 자동 선택) |
| `Danger/danger_detector.py` | `DangerDetector` | 위험 표지 탐지 + 발 위치 기반 진입 판정 |
| `speaker/speaker.py` | `DangerSpeaker` | 음성 경고 출력 (30초 쿨다운, 위험구역 > PPE 우선순위) |
| `ui/safe_eye_danger.py` | — | 통합 웹 모니터 서버 (진입점) |
| `ui/index.html` | — | 브라우저 대시보드 |

### 설계 철학 — 기능별 독립 모듈

센서 · 카메라 · 스피커가 서로의 동작을 막지 않도록 분리. 교체 · 테스트 · 확장이 단위별로 가능하다.

| 모듈 | 핵심 특성 |
|------|----------|
| **IRSensor** | 미감지 시 YOLO 추론 자체를 중단 → 자원 낭비 최소화 |
| **LatestFrameCamera** | 카메라는 항상 ON, 가장 최신 프레임 1장만 저장 → 지연 없는 추론 |
| **DangerSpeaker** | 동일 경고 30초 쿨다운, 위험구역 경고 > PPE 경고 우선순위 |

### Safe Eye Dashboard (Live UI)

운용자가 한 화면에서 모든 안전 지표를 확인:
- 카메라 라이브 피드 + person bbox 오버레이 + 헬멧·조끼 상태 레이블
- 우측 상태 패널 — 감지 인원 · PPE 위반 · 위험 표지 · 추론 FPS · Danger Zone 진입 여부
- 인원수 기반 PPE 분기 모드 표시 (현재 활성 모델: MLC / OD)

---

## 디렉터리 구조

```
safe_eye/
├── ui/                                  # 웹 서버 (진입점)
│   ├── safe_eye_danger.py               #   └ 통합 웹 모니터 서버
│   └── index.html                       #   └ 브라우저 대시보드
│
├── models/                              # 모델 파일
│   ├── yolo11n.pt                       #   └ Person Detector (COCO 사전학습)
│   ├── best.pt                          #   └ PPE OD 모델 (helmet/vest bbox)
│   ├── PPE_MobileNetV3Large.tflite      #   └ PPE 분류 모델 (MLC, 1명용)
│   └── best_p.pt                        #   └ 위험 표지 탐지 모델
│
├── Main/                                # AI 파이프라인 모듈
│   └── detector.py                      #   └ PPEClassifier
│                                        #     (TFLite/PyTorch 자동 선택)
│
├── Danger/                              # 위험 표지판 감지 모듈
│   ├── __init__.py
│   └── danger_detector.py               #   └ DangerDetector
│
├── vision/                              # 카메라 모듈
│   ├── __init__.py
│   └── camera.py                        #   └ LatestFrameCamera
│
├── sensor/                              # IR 센서 모듈
│   ├── __init__.py
│   └── ir_sensor.py                     #   └ IRSensor
│
├── speaker/                             # 음성 경고 모듈
│   ├── __init__.py
│   ├── speaker.py                       #   └ DangerSpeaker
│   └── assets/                          #   └ WAV 파일
│       ├── danger_zone_ko.wav
│       ├── no_helmet_ko.wav
│       └── no_vest_ko.wav
│
├── tools/                               # 테스트 도구
│   └── run_detection_test.py            #   └ 감지 테스트 스크립트
│
└── docs/                                # 문서
```

---

## 모델 구성

| # | 파일 | 기반 | 역할 |
|---|------|------|------|
| 1 | `yolo11n.pt` | YOLOv11n (COCO 사전학습) | Person bbox 감지 |
| 2 | `best.pt` | YOLOv11n (커스텀 학습) | PPE OD (helmet · vest) |
| 3 | `PPE_MobileNetV3Large.tflite` | MobileNetV3Large + Sigmoid×2 | PPE MLC (helmet · vest) |
| 4 | `best_p.pt` | YOLOv11n (커스텀 학습) | 위험 표지 탐지 |

### OD 학습 결과 (YOLOv11n)

| 지표 | 값 |
|------|---|
| Best Epoch | 42 |
| Best mAP50 | 0.923 |
| Last Recall | **0.885** (안전 시스템 특성상 미경고 억제 우선) |
| Last mAP50-95 | 0.785 |
| 학습 / 추론 해상도 | **Image Size 416 학습 / 320 추론** (1.7배 차이로 실시간 동작 경량화) |

### MLC 학습 결과 (MobileNetV3Large)

| 항목 | 값 |
|------|---|
| 백본 | ImageNet 사전학습 가중치 사용 · 마지막 Dense 재학습 |
| 출력 | helmet_prob · vest_prob (sigmoid 2개, 임계값 0.5) |
| Loss / Optimizer | Binary Cross-Entropy / Adam |
| Validation 헬멧 정확도 | **91.0%** |
| Validation 조끼 정확도 | **91.8%** |

> 균등 샘플링 적용 직후 학습이 안정화 → Train·Val 격차 축소, Loss 동반 수렴.

---

## 데이터셋

### OD 학습 데이터

| 항목 | 내용 |
|------|------|
| 출처 | Kaggle HardHat-Vest Dataset · Construction Site Safety (Roboflow) · Roboflow Universe PPE |
| 형식 | YOLO OD (helmet · vest 2클래스, no-vest는 부재 추론) |
| 규모 | **Train 24,941장 / Validation 1,200장** |
| 검수 | 라벨 오류·클래스 불일치 직접 검수 + 추가 검증 데이터 4,000장 품질 보강 |

### MLC 학습 데이터 변환 파이프라인

OD 형식 라벨을 1인 크롭 + 이진 라벨로 자동 변환:

1. **사람 검출 & 크롭** — YOLO로 person bbox 검출 → 사람별 이미지로 분할
2. **겹침 필터** — Containment Ratio 기반 다른 사람 섞인 크롭 제거
3. **PPE 라벨 귀속** — OD bbox 라벨을 각 크롭에 매칭 → helmet · vest 이진 라벨 부여
4. **충돌 제거 & 저장** — 상반된 라벨 제거 → 크롭 이미지 + 이진 라벨 CSV (파일명 기반 상태 코드 11/10/01/00)

### 클래스 불균형 대응 — 균등 샘플링

상태 코드(HV)별 분포가 극단적이라 그대로 학습하면 다수 클래스에 편향:

| 코드 | 의미 | 원본 샘플 수 |
|------|------|-------------|
| 10 | 헬멧 O · 조끼 X | ≈ 25,000장 |
| 11 | 헬멧 O · 조끼 O | ≈ 4,000장 |
| 00 | 헬멧 X · 조끼 X | ≈ 20,000장 |
| **01** | **헬멧 X · 조끼 O** | **≈ 900장 (희소)** |

- 희소 클래스(01·00) → 가용 샘플 **전량 사용**
- 다수 클래스(10·11) → 클래스별 **최대 1,000장**으로 균등 다운샘플링
- Stratified 3-Way 분할: **Train 70% / Validation 15% / Test 15%** · EarlyStopping으로 과적합 방지

> Class Weight 조정·Augmentation 단독 적용은 학습 불안정·과적합 발생 → **데이터 측면 균등화**가 더 효과적이었다.

### 평가용 GT 데이터셋 (11클래스 · OD 학습셋과 분리)

데이터 누수 없는 공정 비교를 위해 별도 구축:

- **전략적 샘플링**: Greedy 알고리즘으로 인원수(1~2명 vs 3인 이상)와 헬멧·조끼 라벨 균형을 동시 만족하는 **원본 100장** 선별
- **데이터 누수 방지**: 테스트셋 원본 100장에서 파생된 모든 크롭 이미지를 MLC·OD 학습셋에서 물리적으로 제거
- **표본**: Few 그룹 (1~2명) **496장 / 588명** + Many 그룹 (3인+) **165장 / 889명** · 총 **1,477명**
- **구간별 정량 평가**: Small / Large 구간 분리 분석 + 혼동행렬 도출 → 미경고·오경고 케이스 정량 확인

---

## 평가 방법론

### 공정 비교의 전제

동일한 **yolo11m person 박스**를 GT·OD·MLC가 공유 → detector 차이로 인한 편향 제거. 두 방식이 같은 평가셋·같은 사람 박스를 보므로 상대 비교 결과는 신뢰할 수 있다.

### GT 생성 규칙

| 항목 | 방식 |
|------|------|
| **헬멧** | Helmet / no_helmet 둘 다 명시 라벨 사용 (3분기, 둘 다 없으면 평가 제외) |
| **조끼** | Vest 명시 라벨만 존재 → **부재 추론** 사용 (Vest bbox 부재 = 미착용) |

> **조끼 한계**: 부재 추론은 라벨 누락·가림 케이스를 미착용으로 오판 가능. 두 방식이 같은 GT를 보므로 **상대 비교는 공정**하나, **절대 정확도는 보수적 해석** 필요.

### PR Curve 평가

미착용 검출을 positive로 잡아 평가:
- **MLC**: sigmoid score sweep → 곡선
- **OD**: 단일 0/1 출력 → PR 평면 위 점 한 개

### Recall 우선

안전 시스템 특성상 **미경고(미착용 → 착용 오판)가 오경고보다 훨씬 위험** → OD는 Recall 우선 추적, MLC는 임계값 0.5로 운영.

### FPS 측정

PC 환경 기준 (yolo11m 222ms · OD 48ms · MLC 55ms/person). Pi5 실측은 향후 과제.

---

## 설치 및 실행

### 요구사항

- Raspberry Pi 5 (4GB+) / Raspberry Pi OS 64-bit (Bookworm)
- Python 3.11
- 주요 의존성: `ultralytics`, `tflite-runtime`, `opencv-python`, `picamera2`, `gpiozero`, `pygame`, `numpy`

### 모델 파일 배치

다음 모델 파일을 `models/` 디렉터리에 배치한다:

```
models/
├── yolo11n.pt                       # Person Detector
├── best.pt                          # PPE OD
├── PPE_MobileNetV3Large.tflite      # PPE MLC
└── best_p.pt                        # 위험 표지 탐지
```

### 실행

```bash
# 1) 저장소 클론
git clone <repository-url>
cd safe_eye

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 통합 웹 모니터 서버 실행 (진입점)
python ui/safe_eye_danger.py
```

브라우저에서 표시되는 대시보드(`ui/index.html`)로 접속하여 실시간 모니터링 화면 확인.

### 감지 테스트

```bash
python tools/run_detection_test.py
```

---

## 데모 시나리오

### 시나리오 1 — 안전모 미착용 감지 및 경고

1. IR 센서가 작업자 접근 감지 → AI 추론 활성화
2. YOLO 사람 감지 → 인원수에 따라 PPE 분류기 자동 선택 (1명: MLC / 2명+: OD)
3. 안전모 미착용 확인 → person bbox 적색 표시, "❌ 헬멧 미착용" 레이블
4. 음성 "안전모를 착용하세요" 출력
5. 30초 쿨다운 후 정상 복귀

### 시나리오 2 — 위험 구역 근접 감지 및 경고

1. 카메라 시야 내 위험 표지판 감지 → 위험 구역 polygon 자동 설정 (30프레임 단위 캐시 갱신)
2. 작업자 발 위치(person bbox 하단 중심점)가 polygon 안으로 진입
3. 즉시 DANGER 경고 발령 → 화면 위험 구역 적색 오버레이
4. 음성 "위험 구역입니다. 즉시 대피하세요" 출력
5. 30초 쿨다운 후 정상 복귀

> 위험구역 경고는 PPE 경고보다 우선순위가 높다.

---

## 주요 결정과 학습

| 키워드 | 내용 |
|-------|------|
| **기획을 현실에 맞게 좁히는 결단** | 이동 로봇 구상에서 고정 카메라 비전 모듈로 범위를 재설계 — '무엇을 빼느냐'가 완성도를 결정했다. |
| **모델만 잘 만든다고 끝이 아니다** | detector·크롭·양자화 미스매치를 잡는 과정에서, **학습-추론 파이프라인의 일치**가 정확도를 좌우함을 체감. 두 모델 비교 시 동일한 yolo11m person 박스 공유로 detector 편향 제거. |
| **비교 실험은 평가 설계가 절반** | GT 자동 생성, 동일 조건 공유, 안전 관점 지표 선택 — 평가 방법론이 모델만큼 중요했다. |
| **OD 학습량이 결과를 좌우** | 동일 데이터·코드에서 OD epoch 차이(20→60)만으로 vest 정확도 9%p 변동 확인. 공정한 모델 비교의 전제 조건. |
| **협업의 현실** | 다른 팀원의 전처리 코드를 정확히 재현하는 일의 어려움, 그리고 그것을 맞췄을 때 결과가 달라지는 것을 경험. |
| **데이터 균형이 학습 안정의 핵심** | Class Weight·Augmentation 단독 적용은 학습 불안정 → 데이터 측면 균등 샘플링이 더 효과적. |

---

## 한계 및 향후 과제

### 기술적 한계

- **조끼 부재 추론**: GT 데이터에 no-vest 라벨이 없어 Vest bbox 부재로만 추론 → 가림·라벨 누락 케이스를 미착용으로 오판 가능
- **다인원 vest 동반 하락**: 두 방식 모두 가림·작은 bbox 영향으로 정확도 저하 (OD 79.6→73.6%, MLC 90.8→77.8%)
- **위험구역 수동 캘리브레이션**: bbox 확장 계수가 설치 환경마다 수동 튜닝 필요
- **카메라 설치 가이드 미정립**: 설치 각도에 따라 발 위치 기반 위험 구역 판정 정확도가 달라짐

### 기술 완성도 — 다음 단계

1. **OD 단일 모델로의 통합** — Epoch 추가 학습으로 vest 정확도 +9.6%p 확인. 두 모델 병행보다 OD 개선에 집중하여 제품화 방향 단순화 (PersonPPE Danger 통합 단일 모델)
2. **다인원 환경 vest 정확도 개선** — 가림·작은 bbox 대응을 위한 데이터 보강 및 다중 프레임 정보 활용 검토
3. **YOLO INT8 양자화 안정화** — Pi5 배포 시 OD 정확도 손실 최소화 — QAT(Quantization-Aware Training) 적용
4. **위험구역 자동 캘리브레이션** — 카메라 설치 시 자동 측정·보정 알고리즘 개발

### 제품화 시 추가 작업

- 다양한 현장 환경 대응 (조명·각도·작업복 색상 다양성 확보 — fine-tuning)
- 위험구역 동적 설정 (관리자가 카메라 뷰에서 직접 영역 지정)
- 관제 시스템 연동 (이벤트 로그·스냅샷 외부 전송, 다중 카메라 대시보드)
- 운영 안정성·산업안전 인증, 다중 카메라 동시 처리 확장

---

## 변경 이력

| 버전 | 내용 |
|------|------|
| v3.0 | 최초 작성 (라인 트레이서 기반) |
| v4.0 | 라인 센서 제거 → 카메라 기반 자율 이동 전환 |
| v4.1 | 초음파 센서 제거 · gpiozero 채택 · 고착 감지 로직 명확화 |
| v5.0 | 전면 재설계 — 이동 기능 전면 제거, 고정 카메라 AI 비전 모듈로 전환. PPE 탐지 CNN vs OD 비교 실험 구조 도입. 위험 구역 판별 방식 변경 (픽셀 거리 → 경고 표지 bbox 확장) |
| **v5.3** | **현재** — 최종 결과보고서 기반 README 정비. **MBC → MLC** 명칭 통일, **하이브리드 파이프라인** 도입(1인 MLC / 다인원 OD), **IR 트리거** 추가, **Safe Eye Dashboard UI** 추가, 실제 디렉터리 구조 반영, KPI 달성 결과 및 정량 평가 결과 정리, 향후 과제로 OD 단일 모델 통합 방향 명시 |

---

## 참고 자료

### 공식 문서
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Raspberry Pi picamera2](https://github.com/raspberrypi/picamera2)
- [MobileNetV3 전이학습 가이드](https://www.tensorflow.org/tutorials/images/transfer_learning)

### 데이터셋
- [Kaggle HardHat-Vest Dataset v3](https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3)
- [Construction Site Safety (Roboflow)](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
- [Roboflow PPE Universe](https://universe.roboflow.com/search?q=class:helmet+and+vest)

### 산업 안전 로봇 사례
- [Cobalt Robotics](https://www.cobaltrobotics.com)

---

*Safe Eye · On-Device AI 산업 현장 안전 감지 시스템 · 2026*
*AI 로봇 SW 개발자 교육과정 · 양시영 · 류상균 · 채윤식 · 박희연*

# 산업 현장 안전 감지 시스템 — 세부 프로젝트 기획서

> **과정명** AI 로봇 SW 개발자 교육과정
> **프로젝트명** On-Device AI 기반 산업 현장 안전 감지 시스템
> **플랫폼** Raspberry Pi 5 + Camera Module
> **작성일** 2026-05-13
> **버전** v5.0

---

## 변경 이력

| 버전 | 주요 변경 내용 |
|------|---------------|
| v3.0 | 최초 작성 (라인 트레이서 기반) |
| v4.0 | 라인 센서 제거 → 카메라 기반 자율 이동 전환 |
| v4.1 | 초음파 센서 제거 / gpiozero 채택 / 고착 감지 로직 명확화 |
| v5.0 | **전면 재설계** — 이동 기능 전면 제거 (바퀴·모터·센서 삭제) / 고정 카메라 기반 AI 비전 모니터링 시스템으로 전환 / PPE 탐지 CNN vs OD 비교 실험 구조 도입 / 위험 구역 판별 방식 변경 (픽셀 거리 → 경고 표지 bbox 확장) / 프로젝트 컨셉 재정립 |

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [기능 정의](#2-기능-정의)
3. [하드웨어 구성](#3-하드웨어-구성)
4. [SW 아키텍처](#4-sw-아키텍처)
5. [AI 모델 설계](#5-ai-모델-설계)
6. [데이터셋 전략](#6-데이터셋-전략)
7. [개발 환경 및 기술 스택](#7-개발-환경-및-기술-스택)
8. [시스템 흐름도](#8-시스템-흐름도)
9. [개발 일정](#9-개발-일정)
10. [리스크 및 대응 방안](#10-리스크-및-대응-방안)
11. [성능 지표 (KPI)](#11-성능-지표-kpi)
12. [미해결 과제 (추후 논의 필요)](#12-미해결-과제)
13. [부록](#부록)

---

## 1. 프로젝트 개요

### 1.1 배경 및 목적

산업 현장에서 안전사고는 매년 800명 이상의 사망자를 낳고 있으며, 사람 감독자만으로는 24시간·전 구역의 안전을 보장하기 어렵다.

Cobalt Robotics, Sentry, Boston Dynamics Spot 등 안전 감시를 목적으로 한 산업용 로봇 도입 사례가 늘어나고 있으며, AI 기반 안전 비전 시스템에 대한 관심과 실제 적용 시도가 빠르게 확산되고 있다.

본 프로젝트는 **On-Device AI를 탑재한 고정형 산업 안전 비전 모듈**의 핵심 기능을 구현한다. 카메라에 들어오는 실시간 이미지를 분석하여 PPE(안전모·안전조끼) 미착용자를 탐지하고, 위험 구역 근접 상황을 판별하여 즉각 경고를 출력한다.

### 1.2 핵심 가치

| 가치 | 설명 |
|------|------|
| **On-Device 추론** | 네트워크 단절 환경에서도 안전 판단 중단 없음 |
| **전용 목적 설계** | 안전 감시 자체가 주 목적인 전용 모듈 |
| **비교 실험 기반** | CNN vs OD 두 접근법 비교를 통한 최적 방식 선택 |
| **즉각 경고** | 위험 감지 후 즉시 음성·LED·화면 경고 출력 |
| **2주 완성** | 핵심 AI 기능에 집중한 현실적인 개발 범위 |

### 1.3 프로젝트 범위

**포함 (In-Scope)**

- 고정 카메라 기반 실시간 영상 분석
- 사람 감지 (YOLO)
- PPE 착용 여부 판별 — CNN 방식 / OD 방식 비교 구현
- 위험 표지 인식 및 위험 구역 근접 판정
- 화면 오버레이 (bbox + 착용 여부 + 경고 표시)
- 음성·LED 경고 출력
- 이벤트 로그 및 스냅샷 저장
- 두 모델 방식의 정량적 성능 비교 평가

**미포함 (Out-of-Scope)**

- 자율 이동 / 장애물 회피 (이동 기능 전면 제거)
- 클라우드 연동 대시보드
- 사람 신원 인식
- 실시간 알람 서버 연동

---

## 2. 기능 정의

### 2.1 기능 목록

| 구분 | ID | 기능명 | 우선순위 |
|------|----|--------|----------|
| 감지 | F-01 | 실시간 사람 감지 (YOLO) | Must |
| 감지 | F-02 | 안전모 착용 여부 판별 — CNN 방식 | Must |
| 감지 | F-03 | 안전조끼 착용 여부 판별 — CNN 방식 | Must |
| 감지 | F-04 | 안전모 착용 여부 판별 — OD 방식 | Must |
| 감지 | F-05 | 안전조끼 착용 여부 판별 — OD 방식 | Must |
| 감지 | F-06 | 위험 표지 인식 + 위험 구역 근접 판정 | Must |
| 표시 | F-07 | 화면 오버레이 (bbox + 상태 레이블) | Must |
| 경고 | F-08 | 음성 경고 출력 | Must |
| 경고 | F-09 | LED 경광등 점멸 | Must |
| 로그 | F-10 | 이벤트 로그 + 스냅샷 저장 | Must |
| 평가 | F-11 | CNN vs OD 정량적 성능 비교 평가 | Must |

### 2.2 결과물 출력 형태

카메라에서 들어오는 실시간 영상 위에 아래 정보를 오버레이하여 화면에 표시한다.

```
┌──────────────────────────────────────────────────────┐
│  [LIVE FEED]                          FPS: 7.2       │
│                                                      │
│  ┌──────────┐  ┌──────────┐                          │
│  │ Person 1 │  │ Person 2 │                          │
│  │ ✅ 헬멧  │  │ ❌ 헬멧  │◄── 적색 bbox            │
│  │ ✅ 조끼  │  │ ✅ 조끼  │                          │
│  └──────────┘  └──────────┘                          │
│                                                      │
│         ⚠ 위험구역 근접 경고 ⚠                      │
│                                                      │
│  [LOG] 14:32:11 — no_helmet detected (P2)            │
└──────────────────────────────────────────────────────┘
```

---

## 3. 하드웨어 구성

### 3.1 하드웨어 구성도

```
┌───────────────────────────────────────────────┐
│              안전 감지 모듈 본체               │
│                                               │
│  ┌────────────────────────────────────────┐   │
│  │         Raspberry Pi 5                 │   │
│  │     (메인 컨트롤 / AI 추론)             │   │
│  └──────┬──────────┬────────────┬─────────┘   │
│         │          │            │             │
│  ┌──────▼──────┐   │   ┌────────▼──────┐      │
│  │ Camera      │   │   │ LED 경광등     │      │
│  │ Module 3    │   │   │ (GPIO PWM)    │      │
│  │ (MIPI CSI)  │   │   └───────────────┘      │
│  └─────────────┘   │                          │
│                    │   ┌───────────────┐      │
│                    └──►│ USB 스피커     │      │
│                        │ (음성 경고)   │      │
│                        └───────────────┘      │
│                                               │
│  ┌────────────────────────────────────────┐   │
│  │  Pi 보조배터리 10,000mAh (USB-C)       │   │
│  └────────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
```

### 3.2 부품 스펙

#### 메인 컴퓨팅

| 항목 | 사양 |
|------|------|
| 보드 | Raspberry Pi 5 (4GB 이상 권장) |
| CPU | Cortex-A76 × 4코어 @ 2.4GHz |
| 저장장치 | MicroSD 32GB 이상 Class10 |
| OS | Raspberry Pi OS 64-bit (Bookworm) |

#### 카메라

| 항목 | 사양 |
|------|------|
| 모델 | Raspberry Pi Camera Module 3 |
| 캡처 해상도 | 640×480 (30fps) |
| 추론 주기 | 5~7fps |
| 인터페이스 | MIPI CSI-2 |
| 설치 형태 | 고정 마운트, 위험 구역이 시야에 들어오도록 배치 |

#### 출력 장치

| 항목 | 사양 |
|------|------|
| LED | GPIO PWM 제어, 상태별 색상·점멸 패턴 |
| 스피커 | USB 연결, pygame.mixer 재생 |
| 디스플레이 | HDMI 연결 모니터 (오버레이 화면 출력) |

#### GPIO 핀 배정 (BCM 기준)

| GPIO | 용도 |
|------|------|
| 12 | LED 경광등 PWM |

---

## 4. SW 아키텍처

### 4.1 레이어 구조

```
╔══════════════════════════════════════════════════════╗
║          Application Layer  (main.py)                ║
║   메인 루프 + 경고 정책 + 쿨다운 관리               ║
╠══════════════════════════════════════════════════════╣
║          AI Inference Layer                          ║
║   person_detector.py — YOLO 사람 감지                ║
║   ppe_cnn.py         — CNN PPE 판별 파이프라인       ║
║   ppe_od.py          — OD PPE 판별 파이프라인        ║
║   danger_detector.py — 위험 표지 + 구역 판별         ║
║   camera.py          — picamera2 프레임 스트림       ║
╠══════════════════════════════════════════════════════╣
║          Output / Display Layer                      ║
║   overlay.py   — 화면 오버레이 렌더링                ║
║   output.py    — LED + 스피커 통합 경고              ║
║   logger.py    — 이벤트 JSON 로그 + 스냅샷           ║
╚══════════════════════════════════════════════════════╝
```

### 4.2 디렉토리 구조

```
safety_monitor/
├── main.py                  # 진입점 + 메인 루프 + 경고 관리
├── config.yaml              # 임계값, 핀 번호, 모델 경로 등
│
├── vision/
│   ├── camera.py            # picamera2 캡처 스트림
│   ├── person_detector.py   # YOLO 사람 감지
│   ├── ppe_cnn.py           # CNN 방식 PPE 판별 파이프라인
│   ├── ppe_od.py            # OD 방식 PPE 판별 파이프라인
│   └── danger_detector.py   # 위험 표지 감지 + 위험 구역 판별
│
├── display/
│   └── overlay.py           # OpenCV 기반 화면 오버레이
│
├── hardware/
│   └── output.py            # LED + 스피커 통합 경고 출력
│
├── logger.py                # 이벤트 JSON 로그 + 스냅샷 저장
│
├── models/                  # TFLite 추론 모델 (Pi5 탑재용)
│   ├── person_detect.tflite
│   ├── ppe_mobilenet.tflite     # CNN 방식용 MobileNetV2
│   ├── ppe_yolo.tflite          # OD 방식용 YOLO
│   └── danger_detect.tflite
│
├── assets/                  # 경고음 WAV
│   ├── no_helmet_ko.wav
│   ├── no_vest_ko.wav
│   └── danger_zone_ko.wav
│
├── training/                # Colab 전용 (Pi5 미탑재)
│   ├── data_prep_cnn.py     # OD 라벨 → CNN 학습 데이터 변환
│   ├── train_cnn.py         # MobileNetV2 학습
│   ├── train_od.py          # YOLO OD 학습
│   ├── convert_tflite.py    # TFLite INT8 변환
│   └── dataset/
│       ├── od/              # OD 학습용 (images/ + labels/ YOLO 형식)
│       └── cnn/             # CNN 학습용 (크롭 이미지 + 이진 라벨 CSV)
│
├── evaluation/
│   ├── generate_gt.py       # OD 라벨 → Ground Truth 자동 생성
│   ├── evaluate_cnn.py      # CNN 방식 성능 평가
│   └── evaluate_od.py       # OD 방식 성능 평가
│
├── logs/
│   └── snapshots/
│
└── tests/
    ├── test_cnn_pipeline.py
    └── test_od_pipeline.py
```

### 4.3 스레드 모델

| 스레드 | 역할 | 실행 주기 |
|--------|------|----------|
| Main Thread | 메인 루프 + 경고 쿨다운 관리 + 정책 판단 | 이벤트 기반 |
| Thread 1: Camera | 프레임 캡처 → 공유 버퍼 | 30fps |
| Thread 2: AI Inference | 모델 실행 → 결과 큐 전달 | 5~7fps |
| Thread 3: Output | LED + 음성 경고 출력 | 이벤트 기반 |
| Thread 4: Logger | JSON 로그 + 스냅샷 저장 | 이벤트 기반 |

### 4.4 경고 관리 정책

이동 기능이 없으므로 상태머신 대신 **경고 쿨다운 기반 관리**를 사용한다.

| 항목 | 내용 |
|------|------|
| 경고 발령 조건 | 위반 감지 3프레임 연속 확인 시 |
| 쿨다운 | 동일 위반 유형에 대해 30초 이내 재경고 억제 |
| 스냅샷 저장 | 새 경고 발령 시마다 저장 |
| 음성 | 위반 유형별 1회 재생 |
| LED | 경고 중 점멸, 쿨다운 종료 후 정상 복귀 |

#### LED 상태별 패턴

| 상태 | 색상 | 점멸 패턴 |
|------|------|---------|
| 정상 모니터링 | 녹색 | 상시 점등 |
| PPE 위반 경고 | 황색 | 고속 점멸 (3Hz) |
| 위험 구역 경고 | 적색 | 고속 점멸 (4Hz) |

---

## 5. AI 모델 설계

### 5.1 모델 구성 요약

| # | 파일명 | 기반 | 역할 | 학습 필요 |
|---|--------|------|------|----------|
| 1 | `person_detect.tflite` | YOLOv12 (COCO) | 사람 bbox 감지 | 불필요 |
| 2 | `ppe_mobilenet.tflite` | MobileNetV2 (FineTuning) | CNN PPE 판별 | 필요 |
| 3 | `ppe_yolo.tflite` | YOLO OD | OD PPE 판별 | 필요 |
| 4 | `danger_detect.tflite` | YOLO OD | 위험 표지 감지 | 필요 |

---

### 5.2 PPE 탐지 — 접근법 1: CNN 방식

#### 파이프라인 개요

```
입력 프레임
    │
    ▼
[Model 1] Person Detector (YOLOv12)
    │  사람 bbox 목록
    ▼
[겹침 필터] Containment Ratio 계산
    │  오염된 crop 제외 → 판정 불가 처리
    ▼
[Crop] 각 person bbox를 개별 이미지로 크롭
    │
    ▼
[Model 2] MobileNetV2 FineTuning
    │  입력: 크롭된 사람 이미지
    │  출력: sigmoid 2개 [helmet_prob, vest_prob]
    ▼
인원 수 + 각 인원의 helmet O/X, vest O/X 결과
```

#### 겹침 필터 (Containment Ratio)

CNN은 한 명만 있는 이미지로 학습하므로, 추론 시 crop에 다른 사람이 걸쳐 있으면 신뢰할 수 없는 결과가 나온다. 아래 기준으로 오염된 crop을 필터링한다.

```
Containment(A가 B를 오염) = 두 bbox 겹치는 넓이 / B bbox 넓이

person A와 person B에 대해:
  - Containment(A→B) > threshold → B의 crop은 판정불가
  - Containment(B→A) > threshold → A의 crop은 판정불가
```

- threshold는 실험으로 결정 (초기값 0.15 권장)
- 판정불가 person은 오버레이에 "⚠ 판정불가"로 표시

#### 모델 출력

| 출력 | 의미 | 판정 기준 |
|------|------|---------|
| `helmet_prob` | 안전모 착용 확률 | ≥ 0.5 → 착용 |
| `vest_prob` | 안전조끼 착용 확률 | ≥ 0.5 → 착용 |

#### CNN 방식의 한계 (명확히 인식)

- YOLO person 감지 단계의 오류가 전체 파이프라인 정확도에 직접 영향
- 학습 데이터가 "단독 인물 크롭"으로만 구성 → 겹친 인물에 대한 일반화 부족
- 두 단계 파이프라인으로 인한 지연 시간 누적

---

### 5.3 PPE 탐지 — 접근법 2: OD 방식

#### 클래스 구성

| 클래스 ID | 클래스명 | 의미 |
|----------|---------|------|
| 0 | `helmet` | 안전모 착용 bbox |
| 1 | `vest` | 안전조끼 착용 bbox |
| 2 | `no-helmet` | 안전모 미착용 bbox |

> `no-vest`는 데이터셋에 클래스 정의는 있으나 라벨이 전무하여 제외. 조끼 미착용은 person bbox 내 vest bbox 부재로 추론한다.

#### 파이프라인 개요

```
입력 프레임
    │
    ▼
[Model 3] YOLO OD (PPE)
    │  출력: helmet / vest / no-helmet bbox 목록
    ▼
[Attribution] PPE bbox → person bbox 귀속
    │  조건: PPE bbox 면적 중 person bbox 내 포함 비율 ≥ 50%
    ▼
각 person에 대한 helmet O/X, vest O/X 판정
```

#### 착용 판정 규칙

| 항목 | 판정 규칙 |
|------|---------|
| 안전모 착용 | person bbox 내 `helmet` bbox 귀속됨 |
| 안전모 미착용 | person bbox 내 `no-helmet` bbox 귀속됨 |
| 안전모 판정불가 | 두 클래스 모두 귀속 없음 |
| 안전조끼 착용 | person bbox 내 `vest` bbox 귀속됨 |
| 안전조끼 미착용 | vest bbox 귀속 없음 (부재 추론) |

#### Person bbox 패딩

PPE bbox와 person bbox의 귀속 정확도를 높이기 위해 person bbox에 일정 비율의 padding을 적용한다. 패딩 비율은 실험으로 결정 (초기값: 상하좌우 각 10%).

---

### 5.4 성능 비교 평가 방법

두 방식을 동일한 테스트셋과 평가 기준으로 비교한다.

#### Ground Truth 자동 생성 (`generate_gt.py`)

기존 OD 라벨에서 이미지 단위 정답을 자동 생성한다.

```
OD 라벨 (이미지별 bbox 목록)
    │
    ▼
person bbox 목록 추출
    │
    ▼
각 person bbox에 대해:
  - 내부에 포함된 helmet/no-helmet bbox 확인 (포함 비율 ≥ 50%)
  - 내부에 포함된 vest bbox 확인
    │
    ▼
이미지 단위 Ground Truth:
  {person_count: N, persons: [{helmet: O/X, vest: O/X}, ...]}
```

#### 평가 지표

| 지표 | 설명 |
|------|------|
| 인원 수 정확도 | 예측 인원 수와 GT 인원 수 일치율 |
| 헬멧 감지 정확도 | 인원 수 일치 케이스에서 헬멧 착용 여부 일치율 |
| 조끼 감지 정확도 | 인원 수 일치 케이스에서 조끼 착용 여부 일치율 |
| FNR (False Negative Rate) | 실제 미착용인데 착용으로 판정한 비율 ← 안전 관점에서 가장 중요 |
| 추론 속도 (fps) | Pi5 실측 기준 |

> **FNR 최소화 우선**: 안전 시스템 특성상 미착용을 착용으로 잘못 판정하는 오류(미경고)가 착용을 미착용으로 잘못 판정하는 오류(오경고)보다 훨씬 위험하다.

---

### 5.5 위험 표지 인식 + 위험 구역 판별

#### 고정 카메라 특성을 활용한 판별 방식

카메라가 고정되어 있어 경고 표지의 화면 내 위치가 일정하다. 픽셀 거리 계산 대신 **경고 표지 bbox를 확장하여 위험 구역 polygon을 생성**하는 방식을 사용한다.

```
[Model 4] Danger Detector → 경고 표지 bbox 감지
    │
    ▼
경고 표지 bbox를 확장 계수로 확장 → 위험 구역 polygon 생성
    │  (확장 계수는 설치 환경 기준으로 1회 튜닝)
    ▼
각 person bbox의 하단 중심점(발 위치) 추출
    │
    ▼
발 위치가 위험 구역 polygon 내에 있는가?
  → YES: DANGER_ZONE 경고
  → NO: 정상
```

#### 발 위치를 기준으로 쓰는 이유

사람의 bbox 하단 중심점은 화면에서 지면 위치를 가장 잘 대표한다. 카메라 각도에 관계없이 "발이 위험 구역 안에 있으면 경고"라는 판단이 중심점 기반보다 안정적이다.

#### 판정 보류 조건

| 조건 | 처리 |
|------|------|
| 경고 표지가 감지되지 않음 | 위험 구역 판정 중단 |
| person bbox 높이 < 프레임 높이의 15% | 너무 멀어 신뢰 불가 → 판정 보류 |

---

### 5.6 추론 속도 목표

| 파이프라인 | Pi5 예상 속도 | 목표 |
|---------|------------|------|
| YOLO(person) + MobileNetV2(CNN) | 5~8fps | ≥ 5fps |
| YOLO(PPE OD 단독) | 8~12fps | ≥ 7fps |

> **목표 FPS: 5fps 이상, 목표치 7fps**
> 안전 모니터링 목적에서 5fps(0.2초 간격 판단)는 충분하다. 사람의 이동 속도 대비 반응 시간이 확보되며, Pi5에서 현실적으로 달성 가능한 수치다.

### 5.7 모델 변환 흐름

```
[Colab] 데이터셋으로 모델 학습 → .keras / .pt 저장
    ↓
[Colab] TFLite INT8 양자화 변환 → .tflite 생성
        (캘리브레이션 이미지 약 100장 — 검증셋에서 샘플링)
    ↓
[scp] .tflite 파일만 Pi5의 models/ 로 복사
    ↓
[Pi5] 각 detector에서 로드 후 추론만 실행
```

---

## 6. 데이터셋 전략

### 6.1 데이터셋 현황

| 항목 | 내용 |
|------|------|
| 원본 이미지 수 | 약 17,000장 |
| 형식 | YOLO OD 형식 (images/ + labels/) |
| 클래스 | 0: Helmet, 1: Vest, 2: No-helmet (3: No-vest는 라벨 전무) |
| 출처 | Kaggle HardHat-Vest Dataset v3, Construction Site Safety, Roboflow Universe |

#### 클래스 분포 (2,600장 처리 기준 추정)

| 클래스 | 샘플 수 | 비율 |
|--------|--------|------|
| Helmet O | 4,483 | 79% |
| Helmet X | 1,185 | 21% |
| Vest O | 485 | 8% |
| Vest X (부재 추론) | 5,283 | 92% |

> Vest O:X 비율이 약 1:11로 매우 불균형. 전체 17,000장 처리 후에도 유사한 비율이 예상됨.

### 6.2 CNN 학습 데이터 생성 파이프라인

OD 형식의 라벨을 CNN 학습용 단일 인물 크롭 + 이진 라벨로 변환한다.

```
OD 라벨 (원본 이미지)
    │
    ▼
YOLOv12로 사람 bbox 감지
    │
    ▼
각 person bbox crop
    │
    ▼
[필터 1] Containment Ratio 체크
    │  다른 person bbox가 이 crop에 15% 이상 포함 → 폐기
    │
    ▼
[필터 2] OD 라벨 기반 PPE 귀속
    │  crop 내 helmet/no-helmet/vest bbox 확인 (포함 비율 ≥ 50%)
    │
    ▼
[필터 3] 라벨 충돌 제거
    │  동일 카테고리에 상반된 라벨 2개 이상 → 폐기
    │  (예: helmet bbox와 no-helmet bbox가 동시에 귀속)
    │
    ▼
최종 학습 데이터:
  crop 이미지 + {helmet: 0/1, vest: 0/1} CSV 라벨
```

### 6.3 OD 학습 데이터

원본 OD 데이터셋을 그대로 사용한다. 클래스는 helmet(0), vest(1), no-helmet(2) 3개.

> `no-vest`가 라벨 없음 → 조끼 미착용 판정은 vest bbox 부재로 추론. 데이터셋 내 vest 라벨이 일관되게 존재하는지 샘플 검증 필요 (→ 미해결 과제 참조).

### 6.4 데이터 불균형 대응

#### Vest O 부족 (1:11) — 복합 대응 필요

| 대응 방법 | 내용 |
|---------|------|
| 추가 데이터 수집 | Roboflow Universe에서 "safety vest", "high visibility vest" 검색 |
| AI 생성 이미지 | 산업 현장 배경 + 조끼 착용 이미지 생성, 실사와 7:3 혼합 사용 |
| 오버샘플링 | Vest O 이미지 반복 샘플링으로 학습 노출 빈도 증가 |
| 강한 Augmentation | Vest O 이미지에 한해 추가 색상·밝기·회전 조합 적용 |
| 클래스 가중치 | 학습 시 vest=1 클래스에 높은 가중치 부여 |

> AI 생성 이미지 주의사항: 실사 이미지 비율 70% 이상 유지. "흰 배경에 조끼 입은 사람" 류의 비현실적 이미지는 제외.

#### Helmet X 부족 (1:3.8) — 클래스 가중치로 대응 가능

| 대응 방법 | 내용 |
|---------|------|
| 클래스 가중치 | helmet=0 클래스에 약 3~4배 가중치 부여 |
| 오버샘플링 | 필요 시 추가 |

### 6.5 데이터 증강 전략

| 기법 | 목적 |
|------|------|
| 밝기 조정 (±30%) | 조명 환경 다양화 |
| 수평 플립 | 좌우 대칭 다양성 확보 |
| 회전 (±15°) | 카메라 각도 대응 |
| 가우시안 블러 (σ=1~2) | 거리·흔들림 대응 |
| HSV 색상 지터 | 조끼 색상 편향 방지 |

---

## 7. 개발 환경 및 기술 스택

| 계층 | 기술 |
|------|------|
| OS | Raspberry Pi OS 64-bit (Bookworm) |
| 언어 | Python 3.11 |
| AI 추론 (Pi5) | tflite-runtime |
| AI 학습 (Colab) | TensorFlow / Keras, Ultralytics YOLO (v12) |
| 컴퓨터 비전 | OpenCV 4.x |
| 카메라 | picamera2 |
| GPIO 제어 | gpiozero (PWMOutputDevice) |
| 음성 출력 | pygame.mixer |
| 버전 관리 | Git + GitHub |

**개발 환경 역할 분리**

| 작업 | 환경 |
|------|------|
| 데이터 전처리 / 라벨 변환 | 개발 PC / Colab |
| CNN 모델 학습 (MobileNetV2) | Colab (GPU) |
| OD 모델 학습 (YOLO) | Colab (GPU) |
| TFLite INT8 변환 | Colab |
| On-Device 추론 / 통합 테스트 | Raspberry Pi 5 |

---

## 8. 시스템 흐름도

### 8.1 메인 추론 루프

```
카메라 프레임 캡처 (30fps 버퍼)
    │
    ▼ (5~7fps 추론 주기)
[Person Detector] YOLOv12
    │  사람 bbox 목록
    ├─────────────────────────────────┐
    ▼                                 ▼
[PPE 판별]                     [Danger 판별]
  CNN 방식 or OD 방식            경고 표지 감지
  helmet / vest 상태             → 위험 구역 polygon
  per person                     → 발 위치 확인
    │                                 │
    └──────────────┬──────────────────┘
                   ▼
           [경고 판단 + 쿨다운 체크]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   [Overlay 렌더링]     [Alert 발령]
   화면에 bbox +        음성 + LED
   상태 레이블 표시      (쿨다운 30초)
         │                   │
         └─────────┬─────────┘
                   ▼
             [Logger]
         이벤트 JSON + 스냅샷
```

### 8.2 CNN PPE 판별 상세 흐름

```
Person bbox 목록
    │
    ▼
Containment Ratio 계산
    │
    ├── 오염 비율 > threshold → 해당 person: "판정불가"
    │
    └── 정상 crop
            │
            ▼
        MobileNetV2 추론
            │
            ▼
        [helmet_prob, vest_prob]
            │
            ├── helmet_prob < 0.5 → 헬멧 미착용 → 경고 대상
            └── vest_prob   < 0.5 → 조끼 미착용 → 경고 대상
```

### 8.3 OD PPE 판별 상세 흐름

```
입력 프레임
    │
    ▼
YOLO OD 추론
    │  bbox: helmet / no-helmet / vest
    │
    ▼
각 PPE bbox × person bbox 포함 비율 계산
    │  포함 비율 ≥ 50% → 해당 person에 귀속
    │
    ├── no-helmet 귀속됨 → 헬멧 미착용
    ├── helmet 귀속 없음 → 안전모 판정불가 (클래스 미감지)
    └── vest 귀속 없음   → 조끼 미착용 (부재 추론)
```

### 8.4 위험 구역 판별 상세 흐름

```
[Danger Detector]
    │  경고 표지 bbox 감지
    │
    ▼
경고 표지 bbox 확장 → 위험 구역 polygon
    │  확장 계수: config에서 조정 (설치 시 1회 튜닝)
    │
    ▼
각 person bbox 하단 중심점 (발 위치) 추출
    │
    ├── person 높이 < 프레임 높이 × 0.15 → 판정 보류 (너무 멀음)
    │
    └── 발 위치 in polygon? → YES: 위험 구역 경고
```

---

## 9. 개발 일정

### 9.1 팀 구성 및 역할

> **총 인원: 4명 / 개발 마감: Day 7 / 현재: Day 3 진행 중**

| 멤버 | 역할 | 현재 담당 |
|------|------|---------|
| 멤버 A | CNN 담당 | PPE CNN 파이프라인, MobileNetV2 학습·변환 |
| 멤버 B | OD 담당 | PPE OD 파이프라인, YOLO 학습·변환 |
| 멤버 C | 데이터 담당 | 데이터 불균형 대응, 학습 데이터 정제 |
| 총괄 (나) | 프로젝트 관리 | 방향 설정, 트러블슈팅, 백업, 미착수 영역 진행 |

> **미착수 영역 (Day 3 기준)**: 위험 구역 판별(danger_detector.py), 화면 오버레이(overlay.py), 전체 통합(main.py)
> 위 세 영역은 총괄이 주도하되, 멤버 C가 데이터 작업 마무리 후 합류하는 구조로 진행한다.

---

### 9.2 잔여 일정 (Day 3 ~ Day 7)

#### Day 3 (오늘, 진행 중) — PPE 학습 준비 수렴 + 미착수 영역 착수

| 멤버 | 작업 | 완료 기준 |
|------|------|---------|
| A | CNN 학습 데이터 최종 정제, MobileNetV2 학습 환경(Colab) 준비 | 학습 데이터셋 확정, Colab 실행 준비 완료 |
| B | OD 학습 데이터 최종 정제, YOLO 학습 환경(Colab) 준비 | 학습 데이터셋 확정, Colab 실행 준비 완료 |
| C | 데이터 불균형 대응 전략 확정 및 적용 (오버샘플링·클래스 가중치·augmentation) | 불균형 대응 적용 완료, 학습 데이터 최종본 확정 |
| 총괄 | camera.py + person_detector.py 구현 확인, overlay.py 설계 시작, danger_detector.py 설계 시작 | Pi5에서 person bbox 출력 확인 |

---

#### Day 4 — 모델 학습 시작 + 오버레이·위험 구역 구현

> A, B는 Colab 학습을 돌리면서 파이프라인 코드를 병행 구현한다.
> 학습은 백그라운드로 돌아가므로 코드 작업과 시간이 겹친다.

| 멤버 | 작업 | 완료 기준 |
|------|------|---------|
| A | Colab MobileNetV2 학습 시작, ppe_cnn.py 구현 (person crop → containment 필터 → 모델 입력) | 학습 실행 확인, 파이프라인 단위 테스트 통과 |
| B | Colab YOLO OD 학습 시작, ppe_od.py 구현 (PPE bbox → person 귀속 로직) | 학습 실행 확인, 파이프라인 단위 테스트 통과 |
| C | overlay.py 구현 (person bbox + 헬멧·조끼 상태 레이블 + 위험 경고 표시) | 샘플 이미지에 오버레이 정상 출력 확인 |
| 총괄 | danger_detector.py 구현 (위험 표지 감지 → bbox 확장 → 발 위치 판정), output.py 구현 | 더미 표지 이미지에서 위험 구역 polygon 생성 확인, 음성·LED 테스트 완료 |

---

#### Day 5 — TFLite 변환 + Pi5 통합

| 멤버 | 작업 | 완료 기준 |
|------|------|---------|
| A | CNN 학습 결과 확인, TFLite INT8 변환, Pi5 추론 속도 실측 | Pi5에서 CNN 파이프라인 fps 기록 |
| B | OD 학습 결과 확인, TFLite 변환, Pi5 추론 속도 실측 | Pi5에서 OD 파이프라인 fps 기록 |
| C | Pi5에서 camera → person detect → overlay 통합 테스트 | 실시간 카메라 피드에 person bbox 오버레이 출력 확인 |
| 총괄 | main.py 메인 루프 구현 (스레드 모델, 경고 쿨다운), logger.py 구현 | 카메라 → 추론 → 오버레이 → 경고 기본 흐름 동작 |

> **Day 5 체크포인트**: 이 시점에 학습 정확도가 기준 미달이면 Day 6 재학습 없이 임계값 조정으로 대응한다. 재학습은 시간상 불가능하다고 전제한다.

---

#### Day 6 — 전체 통합 + 시나리오 테스트

| 멤버 | 작업 | 완료 기준 |
|------|------|---------|
| A + B | 실제 TFLite 모델을 파이프라인에 연결, PPE 판별 정확도 초기 확인, 임계값 조정 | CNN / OD 각각 PPE 위반 → 경고 발령 동작 확인 |
| C | 위험 구역 판별 통합 (danger_detector.py → main.py 연결), 확장 계수 초기 튜닝 | 위험 표지 앞 사람 → 위험 경고 동작 확인 |
| 총괄 | end-to-end 시나리오 테스트, 트러블슈팅 총괄 | 시나리오 1(PPE 위반), 시나리오 2(위험 구역) 각 1회 이상 성공 |

---

#### Day 7 — 안정화 + 데모 준비 + 마감

| 멤버 | 작업 | 완료 기준 |
|------|------|---------|
| A + B | CNN vs OD 정량 비교 (가능한 범위), 최종 모델 확정 (데모에 사용할 방식 결정) | 비교 결과 정리 완료, 최종 모델 Pi5 배포 |
| C | 오버레이 UI 최종 정리, 데모 시나리오 3회 반복 테스트 | 시나리오 3회 중 2회 이상 성공 |
| 총괄 | 코드 정리·주석·README, GitHub 최종 커밋, 발표 자료 방향 확정 | 코드 정리 완료, GitHub push |

---

### 9.3 우선순위 및 범위 축소 기준

시간 내 완성이 어려울 경우 아래 순서로 범위를 축소한다.

| 우선순위 | 기능 | 비고 |
|---------|------|------|
| 1 (Must) | PPE 판별 (헬멧·조끼) + 화면 오버레이 + 음성·LED 경고 | 핵심 데모 기능 |
| 2 (Should) | CNN vs OD 정량 비교 | 시간 여유 시 |
| 3 (Could) | 위험 구역 판별 + 위험 표지 탐지 | Day 6까지 통합 안 되면 데모에서 제외 |
| 제외 가능 | 이벤트 로그 + 스냅샷 저장 | 데모 필수 아님 |

> **위험 구역 기능이 데모에서 제외되더라도**, 구현 시도 과정과 설계는 발표 자료에 포함한다.

---

## 10. 리스크 및 대응 방안

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|---------|
| Pi5 추론 속도 부족 | 중 | 고 | 입력 해상도 축소, 추론 주기 낮춤, 실측 후 최적화 |
| Vest O 데이터 절대 부족 | 고 | 중 | 추가 데이터 수집, AI 생성 이미지, 오버샘플링 복합 적용 |
| no-vest 라벨 품질 불량 | 중 | 중 | 샘플 검증 후 판단 (→ 미해결 과제 참조) |
| CNN 파이프라인 fps 미달 | 중 | 중 | MobileNet 입력 해상도 축소, OD 단독 방식으로 전환 |
| 위험 구역 bbox 확장 계수 부정확 | 중 | 중 | 설치 환경에서 반복 튜닝, 설치 각도 가이드 수립 |
| 겹친 인물이 많은 환경에서 CNN 정확도 저하 | 중 | 중 | containment ratio threshold 실험, OD 방식 보완 |
| 개발 일정 지연 | 중 | 중 | Day 8~9 완충 일정, Must 기능 우선 완성 원칙 |

---

## 11. 성능 지표 (KPI)

| 지표 | 목표값 | 측정 방법 |
|------|--------|---------|
| 헬멧 감지 정확도 | ≥ 80% | 테스트셋 평가 (GT 기준) |
| 조끼 감지 정확도 | ≥ 70% | 테스트셋 평가 (데이터 불균형으로 완화) |
| FNR (미착용 → 착용 오판) | ≤ 15% | 테스트셋 평가 |
| 위험 구역 판별 정확도 | ≥ 80% | 실환경 10회 테스트 |
| 추론 처리 주기 | ≥ 5fps | Pi5 실측 |
| 감지 → 경고 발령 지연 | ≤ 3초 | 스톱워치 측정 |
| 데모 시나리오 성공률 | ≥ 80% | 5회 반복 시나리오 |

---

## 12. 미해결 과제

> 현재 확인 또는 결정이 완료되지 않은 항목. 개발 진행 중 해결 필요.

| # | 과제 | 내용 | 우선순위 |
|---|------|------|---------|
| U-01 | **no-vest 라벨 품질 검증** | vest(1) 라벨이 데이터셋 전체에 빠짐없이 적용되어 있는지 샘플 50장 이상 육안 확인 필요. 누락이 확인되면 "vest 없음 = 미착용" 추론의 신뢰도가 낮아지며, OD 방식의 조끼 판별 전략 재검토 필요 | 높음 |
| U-02 | **Vest O 데이터 불균형 최종 수치 확인** | 17,000장 전체 처리 완료 후 최종 비율 확인 및 오버샘플링·가중치 전략 구체화 | 높음 |
| U-03 | **CNN 겹침 필터 Containment Ratio threshold** | 0.1~0.2 범위에서 실제 데이터로 실험 필요. threshold가 너무 낮으면 정상 crop을 과도하게 판정불가 처리, 너무 높으면 오염된 crop이 통과됨 | 중간 |
| U-04 | **위험 구역 bbox 확장 계수** | 설치 환경과 카메라 각도에 따라 적정 확장 배율이 달라짐. 실제 설치 후 반복 튜닝 필요 | 중간 |
| U-05 | **카메라 설치 높이·각도 기준** | 설치 각도에 따라 발 위치 기반 위험 구역 판정 정확도가 달라짐. 권장 설치 각도(하향 15~30° 등) 가이드 정립 필요 | 중간 |
| U-06 | **OD 방식 조끼 판정 신뢰도** | `no-vest` 클래스 미학습으로 조끼 미착용 판정을 "vest bbox 부재"로만 추론. 가림·각도 등으로 인한 vest 미탐지와 실제 미착용을 구분 불가능한 한계 존재. 운용 임계값 설정으로 부분 보완 | 낮음 |

---

## 부록 A. config.yaml

```yaml
camera:
  width: 640
  height: 480
  fps: 30
  inference_fps: 7

models:
  person_detector:   models/person_detect.tflite
  ppe_cnn:           models/ppe_mobilenet.tflite
  ppe_od:            models/ppe_yolo.tflite
  danger_detector:   models/danger_detect.tflite

thresholds:
  person_confidence:          0.5
  ppe_od_confidence:          0.5
  danger_confidence:          0.5
  cnn_helmet_threshold:       0.5     # sigmoid 출력 헬멧 착용 판정 기준
  cnn_vest_threshold:         0.5     # sigmoid 출력 조끼 착용 판정 기준
  ppe_iou_ratio:              0.5     # PPE bbox → person bbox 귀속 포함 비율
  containment_ratio:          0.15    # CNN crop 오염 판정 기준 (실험값)
  person_min_height_ratio:    0.15    # 위험 구역 판정 최소 사람 크기
  ppe_consecutive_frames:     3       # PPE 위반 확정 연속 프레임 수
  danger_expansion_factor:    2.0     # 경고 표지 bbox 확장 계수 (튜닝 필요)

person_bbox_padding:
  top:    0.10
  bottom: 0.10
  left:   0.10
  right:  0.10

alert:
  cooldown_seconds: 30
  volume: 0.8

led:
  pin:  12
  mode: pwm
```

---

## 부록 B. 데모 시나리오

**시나리오 1 — 안전모 미착용 감지 및 경고**

1. 시스템 시작 → 실시간 오버레이 화면 출력 (녹색 LED 점등)
2. 안전모 미착용 작업자가 카메라 시야에 진입
3. 사람 감지 → PPE 분석 실행
4. 3프레임 연속 `no-helmet` 확인 → 경고 발령
5. 화면 해당 person bbox 적색 표시, "❌ 헬멧 미착용" 레이블
6. 음성 "안전모를 착용하세요" + 황색 LED 고속 점멸
7. 30초 쿨다운 후 정상 복귀

**시나리오 2 — 위험 구역 근접 감지 및 경고**

1. 카메라 시야 내 경고 표지 감지 → 위험 구역 polygon 자동 설정
2. 작업자가 위험 구역 polygon 안으로 진입 (발 위치 기준)
3. 3프레임 연속 확인 → 경고 발령
4. 화면 위험 구역 영역 적색 오버레이, "⚠ 위험구역 근접" 경고 표시
5. 음성 "위험 구역입니다. 즉시 대피하세요" + 적색 LED 고속 점멸
6. 30초 쿨다운 후 정상 복귀

---

## 부록 C. 참고 자료

- TensorFlow Lite 공식 문서: https://www.tensorflow.org/lite/guide
- Raspberry Pi picamera2: https://github.com/raspberrypi/picamera2
- Ultralytics YOLOv12: https://github.com/ultralytics/ultralytics
- Kaggle HardHat-Vest Dataset v3: https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3
- Kaggle Construction Site Safety: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
- Roboflow PPE Universe: https://universe.roboflow.com/search?q=class:helmet+and+vest
- MobileNetV2 전이학습 가이드: https://www.tensorflow.org/tutorials/images/transfer_learning
- Cobalt Robotics (산업 안전 로봇 사례): https://www.cobaltrobotics.com

---

*On-Device AI 프로젝트 기획서 v5.0 | AI 로봇 SW 개발자 교육과정*
*이전 버전(v4.x) 대비 전면 재설계 — 이동 기능 제거, 고정 AI 비전 모듈로 전환*

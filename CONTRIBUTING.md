# 🤝 팀 개발 가이드

> **On-Device AI 산업 현장 안전 감지 로봇** 프로젝트  
> 이 문서는 팀원 전원이 숙지해야 할 **디렉토리 구조**와 **Git 사용 규칙**을 정리한 문서입니다.

---

## 📁 디렉토리 구조

```
safety_robot/
│
├── main.py                  # 진입점 — 상태 머신 루프 실행
├── config.yaml              # 내 PC/라즈베리파이 설정값 (git 제외, 직접 생성)
├── config.example.yaml      # config.yaml 작성 예시 (이걸 복사해서 사용)
├── README.md                # 프로젝트 소개 및 실행 방법
│
├── hardware/                # 하드웨어 제어 (모터, 센서, LED, 스피커)
│   ├── motor.py             # DC 모터 PWM 제어
│   ├── line_sensor.py       # IR 라인 센서 읽기
│   ├── led.py               # LED 경광등 제어
│   └── speaker.py           # 음성 경고 출력
│
├── vision/                  # AI 추론 파이프라인
│   ├── camera.py            # picamera2 프레임 캡처
│   ├── person_detector.py   # 사람 감지 (YOLOv5n TFLite)
│   ├── ppe_classifier.py    # 안전모·조끼 분류 (MobileNetV2 TFLite)
│   └── danger_zone.py       # 위험 표지 감지 + 근접 판정
│
├── core/                    # 상태 관리 및 이벤트 처리
│   ├── event_bus.py         # 스레드 간 이벤트 큐 (Thread-safe)
│   ├── state_machine.py     # IDLE / DETECT / ALERT 상태 관리
│   ├── line_tracer_ctrl.py  # 라인 추종 제어 로직
│   └── alert_manager.py     # 경고 정책 + 쿨다운 관리
│
├── logger/                  # 로그 및 스냅샷 저장
│   ├── event_logger.py      # JSON 형식 이벤트 로그 기록
│   └── snapshot.py          # 위반 발생 시 프레임 이미지 저장
│
├── models/                  # TFLite 모델 파일 (Git 제외)
│   └── .gitkeep             # ← 폴더 유지용, 모델은 Google Drive에서 받을 것
│
├── assets/                  # 경고 음성 WAV 파일
│   ├── no_helmet_ko.wav
│   ├── no_vest_ko.wav
│   └── danger_zone_ko.wav
│
├── logs/                    # 이벤트 로그 + 스냅샷 저장 폴더 (Git 제외)
│   └── .gitkeep
│
└── tests/                   # 단위 테스트
    ├── test_line_sensor.py
    ├── test_ppe_classifier.py
    └── test_event_bus.py
```

---

## ⚠️ 내가 직접 만들어야 하는 파일

아래 파일들은 **Git에서 제외**되어 있어서 Clone해도 없습니다.  
각자 직접 만들어야 합니다.

### config.yaml

`config.example.yaml` 파일을 복사해서 `config.yaml` 로 이름을 바꾼 뒤,  
**본인 라즈베리파이 배선에 맞게 핀 번호를 수정**하세요.

```bash
cp config.example.yaml config.yaml
```

### models/ 폴더 안의 TFLite 파일

모델 파일은 용량 문제로 Google Drive에서 별도 배포합니다.  
👉 https://drive.google.com/drive/folders/1uW6rdyEEV3Fb-kcCuTpG3p4ByJvQeKy3?usp=drive_link

다운받은 파일을 `models/` 폴더에 넣어주세요.

```
models/
├── person_detect.tflite
├── ppe_classify.tflite
└── danger_sign.tflite
```

---

## 🌿 브랜치 구조

```
main
 └── dev                         
      ├── feature/yang
      ├── feature/ryu 
      ├── feature/park   
      └── feature/chae  
```

| 브랜치 | 설명 |
|--------|------|
| `main` | 최종 발표·데모용. 완성된 코드만 올라옴 |
| `dev` | 개발 통합 브랜치. 기능 완성 후 여기에 합침 |
| `feature/xxx` | 각자 맡은 기능 개발 브랜치 |

---

## 🔄 매일 작업 흐름

### 1. 작업 시작 전 — 항상 최신화부터

```bash
# dev 브랜치 최신 내용 받기
git checkout dev
git pull origin dev

# 내 브랜치로 돌아와서 dev 내용 반영
git checkout feature/내브랜치이름
git merge dev
```

> GitHub Desktop: `Current Branch → dev` 로 바꾼 후 상단 **Fetch origin** → **Pull** 클릭  
> 그 다음 내 브랜치로 다시 변경 후 `Branch → Merge into Current Branch → dev` 선택

---

### 2. 작업 후 저장 — 하루에 여러 번 커밋

```bash
git add .
git commit -m "feat: 라인 센서 3채널 읽기 함수 구현"
git push origin feature/내브랜치이름
```

> GitHub Desktop: 좌측 변경 파일 확인 → 하단 커밋 메시지 입력 → **Commit** → 상단 **Push origin**

---

### 3. 기능 완성 후 — PR(Pull Request) 생성

1. GitHub 웹사이트 접속
2. 상단에 뜨는 **"Compare & pull request"** 클릭
3. `feature/내브랜치` → `dev` 방향인지 확인
4. 제목·설명 작성 후 **Create pull request**

---

## ✅ DO / ❌ DON'T

### 해야 할 것

- 작업 시작 전 **항상 `git pull` 로 최신화**
- 커밋은 **작은 단위로 자주** (하루에 여러 번 OK)
- 커밋 메시지는 **명확하게 작성**
- 기능 완성 후 **PR → 팀원 확인 → Merge**
- 충돌 나면 **혼자 해결하려 하지 말고 공유**

### 하면 안 되는 것

- `main` 브랜치에 **직접 push 금지**
- 테스트 안 한 코드를 `dev` 에 **직접 merge 금지**
- 모델 파일(`.tflite`), 로그 파일 **커밋 금지**
- `config.yaml` **커밋 금지** (핀 번호가 사람마다 다름)
- 커밋 메시지를 `"수정"`, `"ㅇㅇ"`, `"asdf"` 같이 **의미없이 쓰기 금지**
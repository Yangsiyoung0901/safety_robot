# 🤝 팀 개발 가이드

> **On-Device AI 산업 현장 안전 감지 로봇** 프로젝트  
> 이 문서는 팀원 전원이 숙지해야 할 **디렉토리 구조**와 **Git 사용 규칙**을 정리한 문서입니다.

---

## 📁 디렉토리 구조

```
safety_robot/
├── main.py              # 진입점 + 상태머신 + 경고 정책
├── config.yaml          # 임계값, 핀 번호 등 전체 설정
│
├── hardware/
│   ├── motor.py         # 모터 PWM 제어 + 자율 이동 로직
│   └── output.py        # LED + 스피커 통합 경고 출력
│
├── vision/
│   ├── camera.py        # picamera2 캡처 스트림
│   └── detector.py      # Person / PPE / Danger 통합 파이프라인
│
├── logger.py            # 이벤트 JSON 로그 + 스냅샷 저장
│
├── models/              # Pi5 탑재용 TFLite 모델 (추론 전용)
│   ├── person_detect.tflite
│   ├── ppe_detect.tflite
│   └── danger_detect.tflite
│
├── assets/              # 경고음 WAV
│   ├── no_helmet_ko.wav
│   ├── no_vest_ko.wav
│   └── danger_zone_ko.wav
│
├── training/            # 학습 코드 (개발 PC / Colab 전용, Pi5 미탑재)
│   ├── train_ppe.py
│   ├── train_danger.py
│   ├── convert_tflite.py
│   └── dataset/
│       ├── ppe/         # images/ + labels/ (YOLO 형식)
│       └── danger/      # images/ + labels/ (YOLO 형식)
│
├── logs/                # 런타임 생성
│   └── snapshots/
│
└── tests/
    ├── test_detector.py
    └── test_motor.py
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
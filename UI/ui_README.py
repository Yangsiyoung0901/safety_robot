# UI 모듈 설명서

# 팀원 코드(safe_eye_monitor.py)의 send_index() 메서드에 들어가는
# 웹 대시보드 HTML 페이지입니다.
# 기존 OpenCV make_panel() 패널을 대체하여 브라우저에서 상태를 표시합니다.

# =============================================================
# 폴더 구조
# =============================================================
#
# safety_robot/
# ├── config.yaml
# ├── vision_README.py
# ├── ui_README.py              ← 이 파일 (UI 설명서)
# ├── ui/
# │   └── index.html            ← 웹 대시보드 페이지
# └── vision/
#     ├── __init__.py
#     └── detector.py

# =============================================================
# index.html이 뭔가
# =============================================================
#
# 라즈베리파이에서 safe_eye_monitor.py를 실행하면
# 웹 서버가 열리고 (기본 http://라즈베리파이IP:8000)
# PC 브라우저에서 접속하면 이 HTML 페이지가 표시된다.
#
# 페이지 구성:
#   ┌──────────────────────────────────────────────────┐
#   │ HEADER: SAFE EYE 로고 + FPS + 감지 인원 수       │
#   ├────────────────────────┬─────────────────────────┤
#   │                        │ Overview                │
#   │                        │  감지 인원 / PPE 위반   │
#   │                        │  위험 표지 / FPS        │
#   │   카메라 실시간 영상   │─────────────────────────│
#   │   (MJPEG 스트림)       │ PPE method              │
#   │                        │  CNN(1명) / OD(2명+)    │
#   │                        │─────────────────────────│
#   │                        │ Person status           │
#   │                        │  #1 Helmet O  Vest O    │
#   │                        │  #2 Helmet O  Vest X    │
#   │                        │  #3 Helmet X  Vest O    │
#   │                        │  #4 Helmet X  Vest X    │
#   └────────────────────────┴─────────────────────────┘

# =============================================================
# 매커니즘
# =============================================================
#
# 1. 카메라 영상 표시
#    - <img src="/stream.mjpg">로 MJPEG 스트림을 받아서 표시
#    - 이건 팀원 코드의 send_stream()이 제공하는 것
#    - HTML에서는 img 태그만 넣으면 자동으로 실시간 영상이 나옴
#
# 2. 상태 정보 갱신
#    - 1초마다 /status 엔드포인트에 fetch 요청
#    - 현재 /status는 텍스트 형식: "OK people=2 ppe=3 signs=1 danger=False fps=5.2"
#    - 이걸 파싱해서 Overview 카드(인원수, FPS 등)를 업데이트
#
# 3. 사람별 PPE 상태 표시
#    - 각 사람마다 카드 1개 표시
#    - 헬멧 O/X, 조끼 O/X를 뱃지로 표시
#    - 착용 완료 = 녹색 뱃지, 미착용 = 빨간 뱃지
#    - 위험 구역 진입 시 = 빨간 테두리 + DANGER 표시
#    - updatePersonList() 함수가 이 카드를 생성
#
# 4. PPE 방식 표시
#    - 사람 1명이면 "CNN — Classification (1명)" 표시
#    - 사람 2명 이상이면 "OD — Object Detection (N명)" 표시
#    - 하이브리드 방식이 어떤 경로를 타는지 실시간으로 확인 가능

# =============================================================
# 적용 방법
# =============================================================
#
# 1단계: send_index()의 HTML 교체
#   - safe_eye_monitor.py에서 send_index() 메서드를 찾는다
#   - page = """...""" 안의 HTML을 index.html 내용으로 통째로 교체
#
# 2단계: /status 엔드포인트 JSON 확장 (사람별 상태 표시를 위해)
#   - 현재 /status는 텍스트만 반환: "OK people=2 ppe=3 ..."
#   - Person Status 카드를 실시간으로 업데이트하려면
#     각 사람의 helmet/vest 정보를 JSON으로 보내야 함
#
#   현재 (텍스트):
#     "OK people=2 ppe=3 signs=1 danger=False fps=5.2"
#
#   변경 후 (JSON):
#     {
#       "people": 2,
#       "ppe": 3,
#       "signs": 1,
#       "danger": false,
#       "fps": 5.2,
#       "persons": [
#         {"num": 1, "has_helmet": true,  "has_vest": true,  "in_danger": false},
#         {"num": 2, "has_helmet": false, "has_vest": true,  "in_danger": false}
#       ]
#     }
#
#   safe_eye_monitor.py에서 수정할 부분:
#     - send_status() 메서드에서 JSON 형식으로 반환
#     - detection_loop()에서 numbered_persons 정보를 전역 변수로 공유
#
# 3단계: 더미 데이터로 먼저 확인
#   - index.html 하단에 주석 처리된 updatePersonList([...]) 부분이 있음
#   - 주석을 해제하면 실제 연동 없이도 카드가 어떻게 보이는지 확인 가능
#   - 확인 끝나면 다시 주석 처리

# =============================================================
# 기존 OpenCV 패널(make_panel)과의 관계
# =============================================================
#
# 기존:
#   - make_panel()이 OpenCV로 360px 패널 이미지를 생성
#   - 카메라 프레임 옆에 np.hstack()으로 붙여서 MJPEG로 전송
#   - 브라우저에서는 합쳐진 이미지 하나만 보임 (상호작용 불가)
#
# 변경 후:
#   - 카메라 프레임만 MJPEG로 전송 (패널 없이)
#   - 브라우저 HTML/CSS가 오른쪽 패널을 직접 렌더링
#   - /status에서 데이터를 받아 실시간 갱신
#   - 장점: 반응형 레이아웃, 깔끔한 디자인, 상호작용 가능
#
# make_panel()과 np.hstack() 관련 코드를 제거하면
# MJPEG 스트림 용량도 줄어들어 FPS가 올라갈 수 있다.

# =============================================================
# 파일 설명
# =============================================================
#
# index.html 구성:
#   - <style>: 다크 테마 CSS (산업용 대시보드 느낌)
#   - <header>: SAFE EYE 로고, LIVE 표시, FPS, 인원 수
#   - <div class="feed-section">: 카메라 MJPEG 스트림
#   - <div class="panel">: 오른쪽 상태 패널
#     - Overview: 감지 인원 / PPE 위반 / 위험 표지 / FPS
#     - Danger Zone: 위험 구역 진입 여부 배너
#     - PPE Method: 현재 CNN/OD 어떤 방식인지
#     - Person Status: 사람별 헬멧·조끼 착용 카드
#   - <script>: 1초마다 /status 폴링 + UI 갱신 함수
# Safe Eye — 작업 보고서

## 목적
- 로컬 카메라 기반 PPE 및 위험 표지 모니터링 서버(safe_eye)의 실행, 스트림 문제 진단 및 정상화

## 요약 결과
- 서버 실행: http://127.0.0.1:8000 (로컬) — `/status` 엔드포인트 응답 확인
- 스트림: MJPEG 스트림 활성화 및 프레임 추출 성공
- 추론: `ultralytics`(YOLO) 미설치로 추론 비활성 — 카메라 전용 폴백 동작

## 수행 작업(핵심)
1. 서버 실행 및 엔드포인트 확인(`/status`, `/stream.mjpg`).
2. `ultralytics` 설치 시도 — 환경 문제로 설치 보류 및 예외 처리 필요 확인.
3. 코드 수정: `safe eye danger.py`
   - `from ultralytics import YOLO` 예외 처리 추가
   - ultralytics/모델 미설치 시에도 카메라 전용 MJPEG 스트림을 지속하도록 camera-only 폴백 루프 추가
   - `JPEG_QUALITY` 및 `CAMERA_WIDTH`/`CAMERA_HEIGHT`를 실험적으로 조정
4. 카메라 하드웨어 확인: `/dev/video0` 존재, OpenCV로 직접 캡처 성공
5. 스트림 캡처 테스트: `curl`로 바이트 캡처 후 첫 JPEG 추출(여러 번 검증)
6. 임시 디버그 파일 정리(모든 `stream_*.bin`, `stream_*.jpg` 등 삭제)

## 변경된 파일
- `safe eye danger.py` — 예외 처리, 카메라 전용 폴백, 설정 변경

## 생성/삭제 파일
- 생성(검증용): 여러 `stream_*.bin`, `stream_*.jpg` (일시적)
- 삭제: 위의 임시 파일들(최종 정리 완료)

## 핵심 로그 발췌
- "Warning: ultralytics package not available — model inference disabled"
- "PPE model not loaded, running camera-only stream."
- 다수의 `GET /status` 및 `GET /stream.mjpg`(200) 요청 기록
(전체 로그 파일: `server.log`)

## 재현 및 주요 명령
- 서버 포그라운드 실행(로그 확인):

```bash
PYTHONUNBUFFERED=1 python3 'safe eye danger.py' --host 127.0.0.1 --port 8000
```

- 스트림 캡처 테스트(예시):

```bash
timeout 4 curl -sS http://127.0.0.1:8000/stream.mjpg > stream_test.bin
# Python으로 첫 JPEG 추출
python3 - <<'PY'
b=open('stream_test.bin','rb').read()
s=b.find(b'\xff\xd8')
e=b.find(b'\xff\xd9',s)
if s!=-1 and e!=-1: open('stream_frame.jpg','wb').write(b[s:e+2])
PY
```

## 권장 후속 작업
1. 추론 활성화: 가상환경에서 `ultralytics` 및 적절한 `torch` 설치 후 모델(`best.pt`) 연결 — 성능/메모리 주의
2. 서비스화: `systemd` 유닛 파일로 자동 실행 설정
3. 품질/대역폭 튜닝: `JPEG_QUALITY`(예: 80~95)와 해상도(720p/1080p) 사이 균형 조정

## 메모
- 코드 변경 파일: `safe eye danger.py`
- 로그 파일: `server.log`


---
보고서 생성일: 2026-05-18

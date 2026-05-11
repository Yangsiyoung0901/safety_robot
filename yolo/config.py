from pathlib import Path  # 파일 경로를 안전하게 다루기 위해 Path를 가져옵니다.

# MODEL_PATH = Path("best_float32.tflite")  # 라즈베리파이에 복사한 YOLO TFLite 모델 파일 경로입니다.

MODEL_PATH = Path("best.pt")  # pt파일

CAMERA_INDEX = 0  # OpenCV에서 사용할 카메라 번호입니다.

CAMERA_WIDTH = 320  # 라즈베리파이 실시간 처리를 위한 카메라 가로 해상도입니다.

CAMERA_HEIGHT = 240  # 라즈베리파이 실시간 처리를 위한 카메라 세로 해상도입니다.

CAMERA_FPS = 30  # 카메라 목표 FPS입니다.

YOLO_IMAGE_SIZE = 416  # YOLO 학습/export 때 사용한 입력 이미지 크기입니다.

CONFIDENCE = 0.35  # YOLO 탐지 결과를 인정할 최소 신뢰도입니다.

IOU = 0.45  # YOLO NMS에서 겹치는 박스를 정리할 IoU 기준입니다.

DEBUG = False  # 화면 디버그 창을 볼지 정합니다.

HEAD_CLASS_ID = 0  # 현재 데이터셋의 head 클래스 번호입니다.

HELMET_CLASS_ID = 1  # 현재 데이터셋의 helmet 클래스 번호입니다.

VEST_CLASS_ID = 2  # 현재 데이터셋의 vest 클래스 번호입니다.

PERSON_CLASS_ID = None  # person 클래스를 학습했다면 이 값을 해당 번호로 바꿉니다.

SAFE_REQUIRED_CLASSES = {HELMET_CLASS_ID, VEST_CLASS_ID}  # safe 판단에 필요한 클래스 번호 집합입니다.

DANGER_IF_HEAD_VISIBLE = False  # head가 보이면 무조건 danger로 볼지 정하는 옵션입니다.

CENTER_MARGIN = 45  # 화면 중앙에서 이 정도 벗어나면 회전하도록 하는 여유 폭입니다.

TARGET_LOST_SEARCH = True  # 탐지 대상이 없을 때 제자리 탐색 회전을 할지 정합니다.

FORWARD_SPEED = 0.38  # 안전하고 중앙에 있을 때 전진 속도입니다.

TURN_SPEED = 0.30  # 대상이 좌우로 벗어났을 때 회전 속도입니다.

SEARCH_SPEED = 0.24  # 대상을 찾지 못했을 때 탐색 회전 속도입니다.

TOO_CLOSE_AREA_RATIO = 0.45  # 대상 박스가 화면의 이 비율 이상을 차지하면 너무 가까운 것으로 봅니다.

DECISION_HISTORY = 5  # 최근 몇 프레임 판단을 모아 최종 판단을 안정화할지 정합니다.

LOG_INTERVAL_SEC = 1.0  # 터미널 로그를 몇 초마다 출력할지 정합니다.

STOP_ON_DANGER = True  # danger 상태에서 로봇을 멈출지 정합니다.

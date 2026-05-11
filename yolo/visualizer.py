import cv2  # 화면에 박스와 글자를 그리기 위해 OpenCV를 가져옵니다.

from config import HEAD_CLASS_ID, HELMET_CLASS_ID, PERSON_CLASS_ID, VEST_CLASS_ID  # 클래스 번호 설정값을 가져옵니다.
from safety_tracker import TargetState  # 추적 상태 타입을 가져옵니다.
from yolo_detector import Detection  # 탐지 결과 타입을 가져옵니다.

LABELS = {HEAD_CLASS_ID: "head", HELMET_CLASS_ID: "helmet", VEST_CLASS_ID: "vest"}  # 기본 클래스 번호와 이름을 매핑합니다.

if PERSON_CLASS_ID is not None:  # person 클래스가 설정되어 있으면 실행합니다.
    LABELS[PERSON_CLASS_ID] = "person"  # person 클래스 이름을 매핑에 추가합니다.

COLORS = {HEAD_CLASS_ID: (0, 0, 255), HELMET_CLASS_ID: (0, 255, 255), VEST_CLASS_ID: (0, 255, 0)}  # 클래스별 표시 색상입니다.


def draw_debug(frame, detections: list[Detection], target_state: TargetState, command: str):  # 디버그 화면을 그리는 함수입니다.
    display = frame.copy()  # 원본 프레임을 훼손하지 않기 위해 복사본을 만듭니다.
    frame_h, frame_w = display.shape[:2]  # 프레임 높이와 너비를 가져옵니다.
    center_x = frame_w // 2  # 화면 중앙 x 좌표를 계산합니다.
    cv2.line(display, (center_x, 0), (center_x, frame_h), (255, 255, 0), 1)  # 화면 중앙선을 그립니다.
    for detection in detections:  # 모든 탐지 결과를 순회합니다.
        color = COLORS.get(detection.class_id, (255, 255, 255))  # 클래스에 맞는 색상을 가져옵니다.
        label = LABELS.get(detection.class_id, str(detection.class_id))  # 클래스 이름을 가져옵니다.
        text = f"{label} {detection.confidence:.2f}"  # 화면에 표시할 라벨 문자열을 만듭니다.
        cv2.rectangle(display, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)  # 탐지 박스를 그립니다.
        cv2.putText(display, text, (detection.x1, max(18, detection.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 박스 위에 라벨을 씁니다.
    if target_state.target_box is not None:  # 대표 추적 대상 박스가 있으면 실행합니다.
        x1, y1, x2, y2 = target_state.target_box  # 대표 대상 박스 좌표를 꺼냅니다.
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)  # 대표 대상 박스를 보라색으로 강조합니다.
    state_color = (0, 255, 0) if target_state.state == "SAFE" else (0, 0, 255)  # SAFE는 초록색, 나머지는 빨간색으로 표시합니다.
    cv2.putText(display, f"{target_state.state} | {command}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)  # 상태와 명령을 표시합니다.
    cv2.putText(display, target_state.reason, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)  # 판단 이유를 표시합니다.
    return display  # 디버그 표시가 끝난 프레임을 반환합니다.

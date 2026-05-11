from config import CENTER_MARGIN, STOP_ON_DANGER, TARGET_LOST_SEARCH  # 주행 판단 설정값을 가져옵니다.
from safety_tracker import TargetState  # 안전 추적 상태 타입을 가져옵니다.


def decide_drive(target_state: TargetState, frame_width: int) -> str:  # 안전 상태와 위치를 모터 명령으로 바꾸는 함수입니다.
    center_x = frame_width // 2  # 화면 중앙 x 좌표를 계산합니다.
    if not target_state.found:  # 탐지 대상이 없으면 실행합니다.
        return "SEARCH" if TARGET_LOST_SEARCH else "STOP"  # 탐색 옵션에 따라 회전 또는 정지를 반환합니다.
    if target_state.state == "TOO_CLOSE":  # 대상이 너무 가까운 경우 실행합니다.
        return "STOP"  # 충돌 방지를 위해 정지합니다.
    if target_state.state == "DANGER" and STOP_ON_DANGER:  # 위험 상태이고 정지 옵션이 켜져 있으면 실행합니다.
        return "STOP"  # 안전을 위해 정지합니다.
    if target_state.cx < center_x - CENTER_MARGIN:  # 대상이 화면 왼쪽에 치우쳐 있으면 실행합니다.
        return "TURN_LEFT"  # 왼쪽으로 회전합니다.
    if target_state.cx > center_x + CENTER_MARGIN:  # 대상이 화면 오른쪽에 치우쳐 있으면 실행합니다.
        return "TURN_RIGHT"  # 오른쪽으로 회전합니다.
    return "FORWARD"  # 대상이 중앙이고 안전하면 전진합니다.

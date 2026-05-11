from collections import Counter, deque  # 최근 판단 기록과 다수결 계산을 위해 가져옵니다.
from dataclasses import dataclass  # 추적 결과를 구조화하기 위해 dataclass를 가져옵니다.

from config import (  # 안전 판단과 주행 판단에 필요한 설정값들을 가져옵니다.
    DANGER_IF_HEAD_VISIBLE,
    DECISION_HISTORY,
    HEAD_CLASS_ID,
    PERSON_CLASS_ID,
    SAFE_REQUIRED_CLASSES,
    TOO_CLOSE_AREA_RATIO,
)
from yolo_detector import Detection  # YOLO 탐지 결과 타입을 가져옵니다.


@dataclass  # TargetState 클래스를 데이터 저장용 클래스로 만듭니다.
class TargetState:  # 현재 프레임에서 추적할 대상과 안전 상태를 담는 클래스입니다.
    found: bool  # 추적할 대상이 발견되었는지 여부입니다.
    state: str  # SAFE, DANGER, SEARCH, TOO_CLOSE 중 하나의 상태입니다.
    cx: int  # 추적 대상의 중심 x 좌표입니다.
    area_ratio: float  # 추적 대상 박스가 화면에서 차지하는 비율입니다.
    reason: str  # 판단 이유를 로그와 디버그 화면에 표시하기 위한 문자열입니다.
    target_box: tuple[int, int, int, int] | None  # 추적 대상 박스 좌표입니다.


class SafetyTracker:  # YOLO 탐지 결과를 로봇 주행용 상태로 바꾸는 클래스입니다.
    def __init__(self) -> None:  # 객체가 생성될 때 한 번 실행되는 초기화 함수입니다.
        self.history = deque(maxlen=DECISION_HISTORY)  # 최근 safe/danger 판단을 저장하는 큐를 만듭니다.

    def choose_target(self, detections: list[Detection]) -> Detection | None:  # 추적할 대표 대상을 고르는 함수입니다.
        if PERSON_CLASS_ID is not None:  # person 클래스를 학습한 경우 실행합니다.
            people = [d for d in detections if d.class_id == PERSON_CLASS_ID]  # person 클래스 탐지만 모읍니다.
            if people:  # person 탐지가 하나라도 있으면 실행합니다.
                return max(people, key=lambda d: d.area)  # 가장 큰 사람 박스를 대표 대상으로 선택합니다.
        safety_parts = [d for d in detections if d.class_id in {HEAD_CLASS_ID, *SAFE_REQUIRED_CLASSES}]  # head, helmet, vest 탐지만 모읍니다.
        if not safety_parts:  # 추적할 안전장비 관련 박스가 없으면 실행합니다.
            return None  # 대상 없음으로 반환합니다.
        return max(safety_parts, key=lambda d: d.area)  # 가장 큰 안전장비 관련 박스를 대표 대상으로 선택합니다.

    def decide_safety(self, detections: list[Detection]) -> tuple[str, str]:  # 탐지 결과로 safe/danger를 판단하는 함수입니다.
        detected_classes = {d.class_id for d in detections}  # 현재 프레임에서 보이는 클래스 번호 집합을 만듭니다.
        has_required = SAFE_REQUIRED_CLASSES.issubset(detected_classes)  # helmet과 vest가 모두 보이는지 확인합니다.
        head_visible = HEAD_CLASS_ID in detected_classes  # head 클래스가 보이는지 확인합니다.
        if DANGER_IF_HEAD_VISIBLE and head_visible:  # head가 보이면 무조건 위험으로 보는 옵션이 켜져 있으면 실행합니다.
            raw_state = "DANGER"  # 즉시 위험 상태로 판단합니다.
            reason = "head visible"  # 판단 이유를 기록합니다.
        elif has_required:  # helmet과 vest가 모두 보이면 실행합니다.
            raw_state = "SAFE"  # 안전 상태로 판단합니다.
            reason = "helmet+vest"  # 판단 이유를 기록합니다.
        else:  # 필요한 장비가 하나라도 없으면 실행합니다.
            raw_state = "DANGER"  # 위험 상태로 판단합니다.
            reason = "missing helmet or vest"  # 판단 이유를 기록합니다.
        self.history.append(raw_state)  # 현재 프레임 판단을 최근 기록에 추가합니다.
        vote = Counter(self.history).most_common(1)[0][0]  # 최근 기록에서 가장 많이 나온 상태를 최종 상태로 고릅니다.
        return vote, reason  # 최종 상태와 판단 이유를 반환합니다.

    def update(self, detections: list[Detection], frame_width: int, frame_height: int) -> TargetState:  # 탐지 결과를 로봇 주행 상태로 바꾸는 함수입니다.
        target = self.choose_target(detections)  # 추적할 대표 대상을 선택합니다.
        if target is None:  # 추적 대상이 없으면 실행합니다.
            self.history.clear()  # 이전 safe/danger 기록을 지워 오래된 판단이 남지 않게 합니다.
            return TargetState(False, "SEARCH", frame_width // 2, 0.0, "no target", None)  # 탐색 상태를 반환합니다.
        area_ratio = target.area / float(frame_width * frame_height)  # 대상 박스가 화면에서 차지하는 비율을 계산합니다.
        target_box = (target.x1, target.y1, target.x2, target.y2)  # 대표 대상 박스 좌표를 튜플로 저장합니다.
        if area_ratio >= TOO_CLOSE_AREA_RATIO:  # 대상이 화면을 너무 크게 차지하면 실행합니다.
            return TargetState(True, "TOO_CLOSE", target.cx, area_ratio, "too close", target_box)  # 너무 가까운 상태를 반환합니다.
        state, reason = self.decide_safety(detections)  # safe/danger 상태를 판단합니다.
        return TargetState(True, state, target.cx, area_ratio, reason, target_box)  # 최종 추적 상태를 반환합니다.

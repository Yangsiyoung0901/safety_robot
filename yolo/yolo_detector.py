from dataclasses import dataclass  # 탐지 결과를 구조화해서 저장하기 위해 dataclass를 가져옵니다.

from ultralytics import YOLO  # YOLO TFLite 모델을 쉽게 실행하기 위해 Ultralytics YOLO 클래스를 가져옵니다.

from config import CONFIDENCE, IOU, MODEL_PATH, YOLO_IMAGE_SIZE  # 모델 추론 설정값을 가져옵니다.


@dataclass  # Detection 클래스를 데이터 저장용 클래스로 만듭니다.
class Detection:  # YOLO 탐지 결과 한 개를 표현하는 클래스입니다.
    class_id: int  # 탐지된 클래스 번호입니다.
    confidence: float  # 탐지 신뢰도입니다.
    x1: int  # 박스 왼쪽 위 x 좌표입니다.
    y1: int  # 박스 왼쪽 위 y 좌표입니다.
    x2: int  # 박스 오른쪽 아래 x 좌표입니다.
    y2: int  # 박스 오른쪽 아래 y 좌표입니다.

    @property  # cx를 속성처럼 쓰기 위해 property를 붙입니다.
    def cx(self) -> int:  # 박스 중심 x 좌표를 계산하는 함수입니다.
        return (self.x1 + self.x2) // 2  # 왼쪽과 오른쪽 좌표의 평균을 반환합니다.

    @property  # cy를 속성처럼 쓰기 위해 property를 붙입니다.
    def cy(self) -> int:  # 박스 중심 y 좌표를 계산하는 함수입니다.
        return (self.y1 + self.y2) // 2  # 위쪽과 아래쪽 좌표의 평균을 반환합니다.

    @property  # area를 속성처럼 쓰기 위해 property를 붙입니다.
    def area(self) -> int:  # 박스 면적을 계산하는 함수입니다.
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)  # 박스 가로와 세로를 곱해서 면적을 반환합니다.


class YoloTfliteDetector:  # YOLO TFLite 모델 로딩과 추론을 담당하는 클래스입니다.
    def __init__(self) -> None:  # 객체가 생성될 때 한 번 실행되는 초기화 함수입니다.
        if not MODEL_PATH.exists():  # 모델 파일이 현재 폴더에 없는지 확인합니다.
            raise FileNotFoundError(f"YOLO TFLite 모델을 찾을 수 없습니다: {MODEL_PATH}")  # 모델이 없으면 오류를 냅니다.
        self.model = YOLO(str(MODEL_PATH))  # Ultralytics로 TFLite 모델을 로드합니다.

    def predict(self, frame) -> list[Detection]:  # 카메라 프레임에서 객체를 탐지하는 함수입니다.
        results = self.model.predict(frame, imgsz=YOLO_IMAGE_SIZE, conf=CONFIDENCE, iou=IOU, verbose=False)  # YOLO 추론을 실행합니다.
        detections: list[Detection] = []  # 반환할 탐지 결과 리스트를 준비합니다.
        for box in results[0].boxes:  # 첫 번째 프레임의 모든 탐지 박스를 순회합니다.
            class_id = int(box.cls[0].item())  # 클래스 번호를 정수로 변환합니다.
            confidence = float(box.conf[0].item())  # 신뢰도를 실수로 변환합니다.
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 박스 좌표를 xyxy 형식으로 가져옵니다.
            detection = Detection(class_id, confidence, int(x1), int(y1), int(x2), int(y2))  # Detection 객체를 만듭니다.
            detections.append(detection)  # 탐지 결과 리스트에 추가합니다.
        return detections  # 모든 탐지 결과를 반환합니다.

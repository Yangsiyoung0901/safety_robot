import cv2  # 카메라 프레임을 읽기 위해 OpenCV를 가져옵니다.

from config import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH  # 카메라 설정값을 가져옵니다.


class Camera:  # OpenCV 카메라 객체를 관리하는 클래스입니다.
    def __init__(self) -> None:  # 객체가 생성될 때 한 번 실행되는 초기화 함수입니다.
        self.cap = cv2.VideoCapture(CAMERA_INDEX)  # 설정된 카메라 번호로 카메라를 엽니다.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)  # 카메라 가로 해상도를 설정합니다.
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)  # 카메라 세로 해상도를 설정합니다.
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)  # 카메라 FPS를 설정합니다.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 카메라 버퍼를 줄여 지연을 낮춥니다.
        if not self.cap.isOpened():  # 카메라가 열리지 않았는지 확인합니다.
            raise RuntimeError("카메라를 열 수 없습니다.")  # 카메라 오류를 알려줍니다.

    def read(self):  # 카메라에서 프레임 한 장을 읽는 함수입니다.
        ok, frame = self.cap.read()  # 프레임 읽기 성공 여부와 프레임을 가져옵니다.
        if not ok:  # 프레임 읽기에 실패했는지 확인합니다.
            raise RuntimeError("카메라 프레임을 읽을 수 없습니다.")  # 프레임 읽기 오류를 알려줍니다.
        return frame  # 읽어온 프레임을 반환합니다.

    def release(self) -> None:  # 카메라 자원을 해제하는 함수입니다.
        self.cap.release()  # OpenCV 카메라 객체를 닫습니다.

# -*- coding: utf-8 -*-

import threading
import time
from typing import Optional, Tuple

import cv2


class LatestFrameCamera:
    """카메라를 계속 켜두고, 가장 최신 프레임 1장만 저장하는 클래스."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps_limit: float = 30.0,
    ) -> None:
        # 수정 포인트:
        # - camera_index: USB 카메라 번호. 보통 0부터 시작.
        # - width/height: 카메라 해상도.
        # - fps_limit: 너무 빠르게 읽지 않도록 제한하는 값.
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps_limit = fps_limit

        # OpenCV 카메라 객체는 start()에서 생성한다.
        self._cap: Optional[cv2.VideoCapture] = None

        # 최신 프레임 1장만 저장한다. 새 프레임이 오면 이전 프레임은 덮어쓴다.
        self._latest_frame = None
        self._latest_time = 0.0

        # 카메라 스레드와 다른 모듈이 동시에 프레임을 읽고 쓰기 때문에 lock을 사용한다.
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """카메라 촬영을 시작한다. 한 번 시작하면 stop() 전까지 계속 동작한다."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._cap = cv2.VideoCapture(self.camera_index)

        # 라즈베리파이 카메라 환경에 맞게 필요한 옵션을 여기서 추가하면 된다.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError("Could not open camera")

        # 실제 프레임 읽기는 백그라운드 스레드에서 계속 실행된다.
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """카메라에서 프레임을 계속 읽는 내부 반복문."""
        min_interval = 1.0 / self.fps_limit if self.fps_limit > 0 else 0.0

        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            ok, frame = self._cap.read() if self._cap is not None else (False, None)
            if ok:
                # 최신 프레임만 보관한다. 영상 전체를 저장하지 않는다.
                with self._lock:
                    self._latest_frame = frame
                    self._latest_time = time.monotonic()
            else:
                time.sleep(0.05)

            spent = time.monotonic() - loop_start
            if min_interval > spent:
                time.sleep(min_interval - spent)

    def get_latest_frame(self):
        """가장 최신 프레임을 복사해서 반환한다. 아직 프레임이 없으면 None."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest(self) -> Tuple[object, float]:
        """최신 프레임과 촬영 시간을 함께 반환한다. 반환값: (frame, timestamp)."""
        with self._lock:
            if self._latest_frame is None:
                return None, self._latest_time
            return self._latest_frame.copy(), self._latest_time

    def stop(self) -> None:
        """카메라 스레드와 카메라 장치를 종료한다."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

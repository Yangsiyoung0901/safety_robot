# -*- coding: utf-8 -*-

import threading
import time
from typing import Callable, Optional


try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    GPIO = None


class IRSensor:
    """IR 센서 값을 계속 읽고, 사람이 감지됐는지 저장하는 클래스."""

    def __init__(
        self,
        pin: int,
        active_high: bool = True,
        poll_interval: float = 0.05,
        callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        # 수정 포인트:
        # - pin: IR 센서가 연결된 라즈베리파이 BCM GPIO 번호.
        # - active_high: 감지 시 GPIO 값이 1이면 True, 감지 시 0이면 False로 바꾼다.
        # - poll_interval: 센서를 몇 초마다 읽을지 설정한다.
        # - callback: 감지 상태가 바뀔 때 실행할 함수.
        self.pin = pin
        self.active_high = active_high
        self.poll_interval = poll_interval
        self.callback = callback

        # 현재 감지 상태. 다른 모듈은 is_detected()로 이 값을 확인한다.
        self._detected = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """IR 센서 읽기를 시작한다. stop() 전까지 백그라운드에서 계속 확인한다."""
        if GPIO is None:
            raise RuntimeError("RPi.GPIO is not available. Run this on Raspberry Pi.")
        if self._thread and self._thread.is_alive():
            return

        # BCM 모드는 GPIO 핀 번호를 기준으로 한다. 보드 물리 핀 번호가 아니다.
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        """GPIO 값을 반복해서 읽는 내부 반복문."""
        previous = None
        while not self._stop_event.is_set():
            raw_value = GPIO.input(self.pin)

            # 센서 종류에 따라 감지 시 1 또는 0이 나올 수 있어서 active_high로 보정한다.
            detected = bool(raw_value) if self.active_high else not bool(raw_value)

            with self._lock:
                self._detected = detected

            # 감지 상태가 바뀐 순간에만 callback을 호출한다.
            if detected != previous and self.callback is not None:
                self.callback(detected)
            previous = detected

            time.sleep(self.poll_interval)

    def is_detected(self) -> bool:
        """현재 사람이 감지됐으면 True, 아니면 False."""
        with self._lock:
            return self._detected

    def stop(self) -> None:
        """IR 센서 스레드와 GPIO 핀 설정을 정리한다."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if GPIO is not None:
            GPIO.cleanup(self.pin)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

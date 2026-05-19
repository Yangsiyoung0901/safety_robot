# -*- coding: utf-8 -*-

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Sequence


class DangerSpeaker:
    """라즈베리파이 기본 오디오 출력으로 안전 경고 음성을 재생하는 클래스."""

    def __init__(
        self,
        message: str = "\uc704\ud5d8\uc9c0\uc5ed\uc785\ub2c8\ub2e4",
        cooldown_seconds: float = 3.0,
        audio_file: Optional[str] = None,
        player_command: Optional[Sequence[str]] = None,
    ) -> None:
        # 수정 포인트:
        # - message: 위험지역 기본 안내 문구.
        # - cooldown_seconds: 같은 경고가 너무 자주 반복되지 않게 막는 시간.
        # - audio_file: "위험지역입니다" 같은 고정 wav 파일을 재생할 때 사용.
        # - player_command: espeak-ng 대신 다른 음성 출력 명령을 쓰고 싶을 때 사용.
        self.message = message
        self.cooldown_seconds = cooldown_seconds
        self.audio_file = Path(audio_file) if audio_file else None
        self.player_command = list(player_command) if player_command else None

        # 경고 종류별 마지막 출력 시간을 저장한다.
        # 예: 같은 사람의 같은 경고는 cooldown_seconds 안에서는 반복 출력하지 않는다.
        self._last_spoken_by_key = {}
        self._lock = threading.Lock()

        # 여러 경고가 동시에 발생해도 음성이 겹치지 않도록 재생은 한 번에 하나씩만 한다.
        self._play_lock = threading.Lock()

    def warn_danger_zone(self, force: bool = False) -> bool:
        """단순히 '위험지역입니다'만 출력할 때 사용한다."""
        return self.speak(self.message, alert_key="danger_zone", force=force)

    def warn_person_status(
        self,
        person_number: int,
        missing_helmet: bool = False,
        missing_vest: bool = False,
        in_danger_zone: bool = False,
        force: bool = False,
    ) -> bool:
        """왼쪽부터 몇 번째 사람인지와 헬멧/조끼/위험지역 상태를 문장으로 출력한다."""
        message = self.make_person_status_message(
            person_number=person_number,
            missing_helmet=missing_helmet,
            missing_vest=missing_vest,
            in_danger_zone=in_danger_zone,
        )
        if not message:
            return False

        alert_key = (
            "person_status",
            person_number,
            missing_helmet,
            missing_vest,
            in_danger_zone,
        )
        return self.speak(message, alert_key=alert_key, force=force)

    def speak(self, message: str, alert_key=None, force: bool = False) -> bool:
        """원하는 문장을 출력한다. 실제 출력이 시작되면 True를 반환한다."""
        now = time.monotonic()
        key = alert_key if alert_key is not None else message
        with self._lock:
            last_spoken = self._last_spoken_by_key.get(key, 0.0)
            if not force and now - last_spoken < self.cooldown_seconds:
                return False
            self._last_spoken_by_key[key] = now

        threading.Thread(target=self._play, args=(message,), daemon=True).start()
        return True

    def make_person_status_message(
        self,
        person_number: int,
        missing_helmet: bool = False,
        missing_vest: bool = False,
        in_danger_zone: bool = False,
    ) -> str:
        """사람 번호와 상태값을 받아 실제로 읽을 한국어 문장을 만든다."""
        alerts = []
        if in_danger_zone:
            alerts.append("\uc704\ud5d8\uc9c0\uc5ed\uc785\ub2c8\ub2e4")
        if missing_helmet and missing_vest:
            alerts.append("\ud5ec\uba67\uacfc \uc870\ub07c\uac00 \uc5c6\uc2b5\ub2c8\ub2e4")
        elif missing_helmet:
            alerts.append("\ud5ec\uba67\uc774 \uc5c6\uc2b5\ub2c8\ub2e4")
        elif missing_vest:
            alerts.append("\uc870\ub07c\uac00 \uc5c6\uc2b5\ub2c8\ub2e4")

        if not alerts:
            return ""

        prefix = f"\uc67c\ucabd\ubd80\ud130 {person_number}\ubc88\uc9f8 \uc0ac\ub78c"
        return f"{prefix}, {'. '.join(alerts)}"

    def _play(self, message: str) -> None:
        """외부 명령을 실행해 음성을 재생한다."""
        command = self._build_command(message)
        if command is None:
            return
        try:
            with self._play_lock:
                subprocess.run(command, check=False)
        except FileNotFoundError:
            print("Audio command not found:", command[0])

    def _build_command(self, message: str):
        """현재 설정에 맞는 음성 출력 명령을 만든다."""
        if self.player_command:
            # player_command 안에 {message}가 있으면 그 위치에 문장을 넣는다.
            # 없으면 명령 맨 뒤에 문장을 추가한다.
            command = [part.replace("{message}", message) for part in self.player_command]
            if "{message}" not in " ".join(self.player_command):
                command.append(message)
            return command

        # audio_file은 고정 안내 음성 파일이 있을 때만 사용한다.
        if self.audio_file and message == self.message:
            return ["aplay", str(self.audio_file)]

        # 기본값은 espeak-ng 한국어 음성 합성이다.
        # 블루투스 스피커가 라즈베리파이 기본 오디오 출력이면 이 소리가 스피커로 나간다.
        return ["espeak-ng", "-v", "ko", message]

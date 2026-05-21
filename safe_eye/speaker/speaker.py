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
        message: str = "위험지역입니다",
        cooldown_seconds: float = 3.0,
        audio_file: Optional[str] = None,
        danger_audio_file: Optional[str] = None,
        ppe_audio_file: Optional[str] = None,
        player_command: Optional[Sequence[str]] = None,
    ) -> None:
        self.message = message
        self.cooldown_seconds = cooldown_seconds

        # 이전 코드 호환: audio_file이 들어오면 danger_audio_file로 사용
        self.danger_audio_file = Path(danger_audio_file or audio_file) if (danger_audio_file or audio_file) else None
        self.ppe_audio_file = Path(ppe_audio_file) if ppe_audio_file else None
        self.player_command = list(player_command) if player_command else None

        self._last_spoken_by_key = {}
        self._lock = threading.Lock()
        self._play_lock = threading.Lock()

    def warn_danger_zone(self, force: bool = False) -> bool:
        """위험 구역 경고음을 출력한다."""
        return self.speak(
            self.message,
            alert_key="danger_zone",
            force=force,
            audio_file=self.danger_audio_file,
        )

    def warn_person_status(
        self,
        person_number: int,
        missing_helmet: bool = False,
        missing_vest: bool = False,
        in_danger_zone: bool = False,
        force: bool = False,
    ) -> bool:
        """사람별 헬멧/조끼/위험지역 상태 경고를 출력한다."""
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

        # PPE 위반이 있으면 PPEWarning.wav 사용
        # 위험구역만 있으면 DangerWarning.wav 사용
        if missing_helmet or missing_vest:
            audio_file = self.ppe_audio_file
        elif in_danger_zone:
            audio_file = self.danger_audio_file
        else:
            audio_file = None

        return self.speak(
            message,
            alert_key=alert_key,
            force=force,
            audio_file=audio_file,
        )

    def speak(
        self,
        message: str,
        alert_key=None,
        force: bool = False,
        audio_file: Optional[Path] = None,
    ) -> bool:
        """원하는 문장 또는 지정된 WAV 파일을 출력한다."""
        now = time.monotonic()
        key = alert_key if alert_key is not None else message

        with self._lock:
            last_spoken = self._last_spoken_by_key.get(key, 0.0)
            if not force and now - last_spoken < self.cooldown_seconds:
                return False
            self._last_spoken_by_key[key] = now

        threading.Thread(
            target=self._play,
            args=(message, audio_file),
            daemon=True,
        ).start()
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
            alerts.append("위험지역입니다")

        if missing_helmet and missing_vest:
            alerts.append("헬멧과 조끼가 없습니다")
        elif missing_helmet:
            alerts.append("헬멧이 없습니다")
        elif missing_vest:
            alerts.append("조끼가 없습니다")

        if not alerts:
            return ""

        prefix = f"왼쪽부터 {person_number}번째 사람"
        return f"{prefix}, {'. '.join(alerts)}"

    def _play(self, message: str, audio_file: Optional[Path] = None) -> None:
        """외부 명령을 실행해 음성을 재생한다."""
        command = self._build_command(message, audio_file)
        if command is None:
            return

        try:
            with self._play_lock:
                subprocess.run(command, check=False)
        except FileNotFoundError:
            print("Audio command not found:", command[0])

    def _build_command(self, message: str, audio_file: Optional[Path] = None):
        """현재 설정에 맞는 음성 출력 명령을 만든다."""
        if audio_file and audio_file.exists() and audio_file.stat().st_size > 0:
            return ["aplay", str(audio_file)]

        if self.player_command:
            command = [part.replace("{message}", message) for part in self.player_command]
            if "{message}" not in " ".join(self.player_command):
                command.append(message)
            return command

        return ["espeak-ng", "-v", "ko", message]
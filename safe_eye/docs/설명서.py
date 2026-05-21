# -*- coding: utf-8 -*-

import time

from hardware import DangerSpeaker, IRSensor, LatestFrameCamera

# 다른 코드에서는 from hardware import DangerSpeaker, IRSensor, LatestFrameCamera 
# 하면 hardware 폴더 안에있는 DangerSpeaker, IRSensor, LatestFrameCamera 클래스를
# 불러와서 쓸수있습니다. __init__.py 이건 다른 파일에서 불러올때 편하도록 클래스명 모아논거라
# 삭제하면 귀찮아질지도

# 매커니즘:
# 메인 코드에서 카메라, IR 센서 실행
# 스피커는 전원이 켜진 상태이고 라즈베리파이와 블루투스로 연결되어 있음
#
# -> 카메라는 계속 촬영하면서 마지막 1프레임만 덮어쓰기 방식으로 저장
# -> IR 센서에서 사람이 감지되면 마지막 1프레임을 불러옴
# -> 불러온 프레임을 AI/YOLO 모듈에 전달
# -> YOLO 모듈에서 사람, 헬멧, 조끼, 위험구역을 감지

# -> 감지된 사람들을 화면 왼쪽부터 정렬해서 번호를 부여
# -> 각 사람마다 헬멧 착용 여부, 조끼 착용 여부, 위험구역 진입 여부를 판단
# -> 헬멧이 없거나 조끼가 없거나 위험구역에 들어간 사람이 있으면 스피커 모듈 호출
# -> 스피커 모듈에서 "왼쪽부터 n번째 사람, 헬멧이 없습니다" 같은 안내 문장 생성
# -> 라즈베리파이 기본 오디오 출력으로 음성 재생
# -> 기본 오디오 출력이 블루투스 스피커이면 블루투스 스피커에서 안내 음성이 출력됨
# -> 같은 경고가 너무 자주 반복되지 않도록 일정 시간 쿨다운 적용
# -> 다시 IR 센서 감지와 최신 프레임 확인을 반복

# 카메라 구동을 이렇게 하는이유 : ir센서 감지하고 즉시 켜야 사람이 지나가기전에 판단하는데 
# 감지하고 카메라키고 시작하면 사람이 지나가버릴수도 있어서.

# 1. 카메라를 새로 켜는 시간을 없애기 위해
# 2. IR 센서 감지 순간에 가장 가까운 프레임을 바로 쓰기 위해
# 3. YOLO 분석이 느려도 카메라는 계속 최신 장면을 유지하기 위해
# 4. 영상 전체를 저장하지 않고 메모리를 아끼기 위해


IR_PIN = 17


def on_ir_change(detected: bool) -> None:
    # IR 센서 감지 상태가 바뀔 때마다 실행되는 함수.
    # 실제 프로젝트에서는 여기서 로그를 남기거나 상태 표시 LED를 켤 수 있다.
    print("IR person detected:", detected)


def main() -> None:
    # 수정 포인트:
    # - camera_index: 카메라 번호.
    # - IR_PIN: IR 센서가 연결된 GPIO 번호.
    # - DangerSpeaker(): 기본 오디오 출력으로 음성을 내보낸다.
    camera = LatestFrameCamera(camera_index=0, width=640, height=480)
    ir_sensor = IRSensor(pin=IR_PIN, active_high=True, callback=on_ir_change)
    speaker = DangerSpeaker()

    # start()를 호출하면 카메라와 IR 센서는 stop() 전까지 계속 동작한다.
    camera.start()
    ir_sensor.start()

    try:
        while True:
            # 카메라는 계속 촬영 중이고, 여기서는 가장 최신 1프레임만 가져온다.
            frame = camera.get_latest_frame()

            # IR 센서에 사람이 감지된 경우에만 AI 모듈에 프레임을 넘기는 예시.
            # 항상 YOLO를 돌리고 싶으면 "and ir_sensor.is_detected()" 조건을 제거하면 된다.
            if frame is not None and ir_sensor.is_detected():
                # Pass frame to the AI/software module here.
                # Example:
                # result = danger_module.check(frame)
                # for person in result.persons:
                #     speaker.warn_person_status(
                #         person_number=person.num,
                #         missing_helmet=not person.has_helmet,
                #         missing_vest=not person.has_vest,
                #         in_danger_zone=person.in_danger,
                #     )
                pass

            # CPU를 너무 많이 쓰지 않도록 짧게 쉰다.
            time.sleep(0.03)
    finally:
        # 프로그램이 종료될 때 하드웨어 자원을 정리한다.
        ir_sensor.stop()
        camera.stop()


if __name__ == "__main__":
    main()

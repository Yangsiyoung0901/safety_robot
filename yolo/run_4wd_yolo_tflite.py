import time  # 로그 출력 간격을 계산하기 위해 time 모듈을 가져옵니다.

import cv2  # 디버그 화면 표시와 키 입력 처리를 위해 OpenCV를 가져옵니다.

from camera import Camera  # 카메라 입력 클래스를 가져옵니다.
from config import DEBUG, LOG_INTERVAL_SEC  # 실행 설정값을 가져옵니다.
from drive_decision import decide_drive  # 안전 상태를 주행 명령으로 바꾸는 함수를 가져옵니다.
from motor_4wd import Motor4WD  # 4륜 모터 제어 클래스를 가져옵니다.
from safety_tracker import SafetyTracker  # 안전장비 탐지 기반 추적 클래스를 가져옵니다.
from visualizer import draw_debug  # 디버그 화면 그리기 함수를 가져옵니다.
from yolo_detector import YoloTfliteDetector  # YOLO TFLite 탐지 클래스를 가져옵니다.


def main() -> None:  # 4륜 로봇 실행 메인 함수입니다.
    camera = Camera()  # 카메라를 엽니다.
    detector = YoloTfliteDetector()  # YOLO TFLite 모델을 로드합니다.
    tracker = SafetyTracker()  # 탐지 결과를 안전 상태로 바꾸는 추적기를 만듭니다.
    motors = Motor4WD()  # 4륜 모터 제어 객체를 만듭니다.
    prev_log_time = 0.0  # 마지막 로그 출력 시간을 저장할 변수를 초기화합니다.
    try:  # 실행 중 예외가 나도 모터를 정지시키기 위한 try 블록입니다.
        while True:  # 사용자가 종료할 때까지 계속 반복합니다.
            frame = camera.read()  # 카메라 프레임을 읽습니다.
            frame_h, frame_w = frame.shape[:2]  # 프레임 크기를 가져옵니다.
            detections = detector.predict(frame)  # YOLO로 head, helmet, vest 또는 person을 탐지합니다.
            target_state = tracker.update(detections, frame_w, frame_h)  # 탐지 결과를 추적/안전 상태로 변환합니다.
            command = decide_drive(target_state, frame_w)  # 안전 상태와 위치를 모터 명령으로 변환합니다.
            motors.apply(command)  # 4륜 모터에 명령을 적용합니다.
            now = time.time()  # 현재 시간을 가져옵니다.
            if now - prev_log_time >= LOG_INTERVAL_SEC:  # 로그 출력 간격이 지났는지 확인합니다.
                prev_log_time = now  # 마지막 로그 출력 시간을 갱신합니다.
                print(f"state={target_state.state} command={command} reason={target_state.reason} detections={len(detections)}")  # 상태 로그를 출력합니다.
            if DEBUG:  # 디버그 화면이 켜져 있으면 실행합니다.
                display = draw_debug(frame, detections, target_state, command)  # 디버그 화면을 그립니다.
                cv2.imshow("4WD YOLO TFLite Robot", display)  # 디버그 창을 표시합니다.
                if cv2.waitKey(1) & 0xFF == ord("q"):  # q 키가 눌렸는지 확인합니다.
                    break  # 반복문을 종료합니다.
    except KeyboardInterrupt:  # 사용자가 Ctrl+C로 종료하면 실행합니다.
        print("사용자 종료")  # 종료 메시지를 출력합니다.
    finally:  # 정상 종료와 오류 종료 모두에서 실행합니다.
        motors.cleanup()  # 모터와 부저를 정지하고 정리합니다.
        camera.release()  # 카메라를 닫습니다.
        if DEBUG:  # 디버그 창이 켜져 있었으면 실행합니다.
            cv2.destroyAllWindows()  # OpenCV 창을 모두 닫습니다.
        print("4륜 로봇 정지 완료")  # 종료 완료 메시지를 출력합니다.


if __name__ == "__main__":  # 이 파일을 직접 실행할 때만 실행합니다.
    main()  # 메인 함수를 호출합니다.

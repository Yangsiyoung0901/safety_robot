# robot_camera_motion_safe_drive_4wd.py

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from gpiozero import Motor, PWMOutputDevice, Buzzer

# ============================================================
# [1] 기본 설정
# ============================================================

MODEL_PATH = "safe_danger_model.tflite"
IMG_SIZE = 224
CAMERA_INDEX = 0

DEBUG = True

SAFE_THRESHOLD = 0.5

# ============================================================
# [2] 움직임 감지 설정
# ============================================================

MIN_AREA = 2500
TOO_CLOSE_AREA = 90000
CENTER_MARGIN = 80

# ============================================================
# [3] 속도 설정
# ============================================================

FORWARD_SPEED = 0.40
TURN_SPEED = 0.32
SEARCH_SPEED = 0.25

# ============================================================
# [4] GPIO 핀 설정 - 4륜 기준
# ============================================================
# BCM 번호 기준
#
# LF = Left Front  = 왼쪽 앞 모터
# LR = Left Rear   = 왼쪽 뒤 모터
# RF = Right Front = 오른쪽 앞 모터
# RR = Right Rear  = 오른쪽 뒤 모터

# 왼쪽 앞 모터
LF_IN1 = 17
LF_IN2 = 27

# 왼쪽 뒤 모터
LR_IN1 = 5
LR_IN2 = 6

# 오른쪽 앞 모터
RF_IN1 = 22
RF_IN2 = 23

# 오른쪽 뒤 모터
RR_IN1 = 24
RR_IN2 = 25

# 좌우 속도 제어 PWM
# 모터드라이버가 2개라면 왼쪽 그룹 PWM, 오른쪽 그룹 PWM으로 사용
LEFT_PWM = 18
RIGHT_PWM = 13

# 부저
BUZZER_PIN = 26

# ============================================================
# [5] 모터 / PWM / 부저 객체 생성
# ============================================================

left_front_motor = Motor(forward=LF_IN1, backward=LF_IN2)
left_rear_motor = Motor(forward=LR_IN1, backward=LR_IN2)

right_front_motor = Motor(forward=RF_IN1, backward=RF_IN2)
right_rear_motor = Motor(forward=RR_IN1, backward=RR_IN2)

left_pwm = PWMOutputDevice(LEFT_PWM)
right_pwm = PWMOutputDevice(RIGHT_PWM)

buzzer = Buzzer(BUZZER_PIN)

# ============================================================
# [6] 모터 제어 함수
# ============================================================

def set_speed(left_speed, right_speed=None):
    """
    좌우 모터 그룹 속도 설정.
    right_speed를 생략하면 좌우 같은 속도로 설정함.
    """
    if right_speed is None:
        right_speed = left_speed

    left_speed = max(0.0, min(1.0, left_speed))
    right_speed = max(0.0, min(1.0, right_speed))

    left_pwm.value = left_speed
    right_pwm.value = right_speed


def left_group_forward():
    """
    왼쪽 앞/뒤 모터를 모두 앞으로 회전.
    """
    left_front_motor.forward()
    left_rear_motor.forward()


def left_group_backward():
    """
    왼쪽 앞/뒤 모터를 모두 뒤로 회전.
    """
    left_front_motor.backward()
    left_rear_motor.backward()


def right_group_forward():
    """
    오른쪽 앞/뒤 모터를 모두 앞으로 회전.
    """
    right_front_motor.forward()
    right_rear_motor.forward()


def right_group_backward():
    """
    오른쪽 앞/뒤 모터를 모두 뒤로 회전.
    """
    right_front_motor.backward()
    right_rear_motor.backward()


def forward(speed=FORWARD_SPEED):
    """
    4륜 전진.
    네 개 모터 모두 앞으로 회전.
    """
    set_speed(speed)
    left_group_forward()
    right_group_forward()


def stop():
    """
    4륜 정지.
    모든 모터 정지 + PWM 0.
    """
    left_front_motor.stop()
    left_rear_motor.stop()
    right_front_motor.stop()
    right_rear_motor.stop()

    left_pwm.value = 0
    right_pwm.value = 0


def turn_left(speed=TURN_SPEED):
    """
    제자리 좌회전.
    왼쪽 바퀴 2개는 뒤로,
    오른쪽 바퀴 2개는 앞으로 회전.
    """
    set_speed(speed)
    left_group_backward()
    right_group_forward()


def turn_right(speed=TURN_SPEED):
    """
    제자리 우회전.
    왼쪽 바퀴 2개는 앞으로,
    오른쪽 바퀴 2개는 뒤로 회전.
    """
    set_speed(speed)
    left_group_forward()
    right_group_backward()


def search_turn(speed=SEARCH_SPEED):
    """
    움직이는 물체가 없을 때 탐색 회전.
    천천히 오른쪽으로 제자리 회전.
    """
    set_speed(speed)
    left_group_forward()
    right_group_backward()


# ============================================================
# [7] 부저 제어 함수
# ============================================================

def alarm_on():
    """
    danger 감지 시 부저 켜기.
    """
    buzzer.on()


def alarm_off():
    """
    safe 상태 또는 일반 이동 시 부저 끄기.
    """
    buzzer.off()


# ============================================================
# [8] TFLite 모델 로드
# ============================================================

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# ============================================================
# [9] safe / danger 판별 함수
# ============================================================

def predict_safe_danger(frame):
    """
    카메라 프레임을 safe/danger로 분류.
    """

    # OpenCV는 BGR이므로 모델 입력용 RGB로 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 모델 입력 크기 224x224로 변환
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    # 배치 차원 추가
    input_data = np.expand_dims(img, axis=0)

    # 모델 입력 타입 확인
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32)
    else:
        input_data = input_data.astype(input_dtype)

    # 모델에 입력 넣기
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # 추론 실행
    interpreter.invoke()

    # 출력 가져오기
    output = interpreter.get_tensor(output_details[0]["index"])

    # sigmoid 출력값 기준
    score = float(output[0][0])

    if score >= SAFE_THRESHOLD:
        label = "safe"
        confidence = score
    else:
        label = "danger"
        confidence = 1.0 - score

    return label, confidence


# ============================================================
# [10] 움직이는 물체 감지 함수
# ============================================================

def detect_motion(frame, back_subtractor):
    """
    카메라 영상에서 움직이는 물체를 감지.
    사람을 직접 인식하는 것이 아니라,
    배경과 달라진 움직임 영역을 찾는 방식.
    """

    # 배경 제거
    fg_mask = back_subtractor.apply(frame)

    # 노이즈 완화
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # 움직임 영역을 흰색으로 분리
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 작은 노이즈 제거 및 영역 보정
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    # 외곽선 찾기
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, thresh

    # 가장 큰 움직임 영역 선택
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # 너무 작은 움직임은 노이즈로 판단
    if area < MIN_AREA:
        return None, thresh

    # 사각형 정보 계산
    x, y, w, h = cv2.boundingRect(largest)

    # 중심점 계산
    cx = x + w // 2
    cy = y + h // 2

    motion_info = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "area": area
    }

    return motion_info, thresh


# ============================================================
# [11] 카메라 열기
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    stop()
    alarm_off()
    exit()

# ============================================================
# [12] 배경 제거기 생성
# ============================================================

back_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=120,
    varThreshold=50,
    detectShadows=True
)

print("4륜 카메라 기반 자율 구동 시작")

# ============================================================
# [13] 메인 루프
# ============================================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            stop()
            alarm_off()
            break

        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2

        # 움직이는 물체 감지
        motion_info, motion_mask = detect_motion(frame, back_subtractor)

        # safe/danger AI 판별
        label, confidence = predict_safe_danger(frame)

        robot_state = "STOP"

        # ====================================================
        # [14] 자율 구동 판단
        # ====================================================

        if motion_info is None:
            # 움직이는 대상이 없으면 천천히 회전하며 탐색
            alarm_off()
            search_turn(SEARCH_SPEED)
            robot_state = "SEARCH"

        else:
            cx = motion_info["cx"]
            area = motion_info["area"]

            if area >= TOO_CLOSE_AREA:
                # 너무 가까우면 충돌 위험이 있으므로 정지
                alarm_off()
                stop()
                robot_state = "TOO CLOSE STOP"

            elif label == "danger":
                # danger면 정지 + 부저
                stop()
                alarm_on()
                robot_state = "DANGER STOP + BUZZER"

            elif cx < center_x - CENTER_MARGIN:
                # 대상이 화면 왼쪽이면 좌회전
                alarm_off()
                turn_left(TURN_SPEED)
                robot_state = "TURN LEFT"

            elif cx > center_x + CENTER_MARGIN:
                # 대상이 화면 오른쪽이면 우회전
                alarm_off()
                turn_right(TURN_SPEED)
                robot_state = "TURN RIGHT"

            else:
                # 대상이 중앙이고 danger가 아니면 전진
                alarm_off()
                forward(FORWARD_SPEED)
                robot_state = "FORWARD"

        # ====================================================
        # [15] 디버그 화면 표시
        # ====================================================

        if DEBUG:
            display = frame.copy()

            # 중앙 기준선
            cv2.line(
                display,
                (center_x, 0),
                (center_x, frame_h),
                (255, 255, 0),
                2
            )

            # 중앙 허용 범위 왼쪽
            cv2.line(
                display,
                (center_x - CENTER_MARGIN, 0),
                (center_x - CENTER_MARGIN, frame_h),
                (100, 100, 255),
                1
            )

            # 중앙 허용 범위 오른쪽
            cv2.line(
                display,
                (center_x + CENTER_MARGIN, 0),
                (center_x + CENTER_MARGIN, frame_h),
                (100, 100, 255),
                1
            )

            if motion_info is not None:
                x = motion_info["x"]
                y = motion_info["y"]
                w = motion_info["w"]
                h = motion_info["h"]
                cx = motion_info["cx"]
                cy = motion_info["cy"]
                area = motion_info["area"]

                # 움직임 영역 박스
                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

                # 움직임 중심점
                cv2.circle(
                    display,
                    (cx, cy),
                    6,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    display,
                    f"Motion Area: {int(area)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

            text1 = f"AI: {label.upper()} {confidence * 100:.1f}%"
            text2 = f"ROBOT: {robot_state}"

            cv2.putText(
                display,
                text1,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if label == "safe" else (0, 0, 255),
                2
            )

            cv2.putText(
                display,
                text2,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.imshow("4WD Robot Camera Motion Safe Drive", display)
            cv2.imshow("Motion Mask", motion_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop()
                alarm_off()
                break

except KeyboardInterrupt:
    print("강제 종료됨")
    stop()
    alarm_off()

finally:
    stop()
    alarm_off()
    cap.release()

    if DEBUG:
        cv2.destroyAllWindows()

    print("4륜 로봇 정지 완료")
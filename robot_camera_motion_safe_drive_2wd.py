# robot_camera_motion_safe_drive_2wd.py

import cv2
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter
from gpiozero import Motor, PWMOutputDevice, Buzzer

# =========================
# 설정
# =========================

MODEL_PATH = "safe_danger_model.tflite"

IMG_SIZE = 224
CAMERA_INDEX = 0

DEBUG = True   # 라즈베리파이 화면 없이 구동할 때는 False

SAFE_THRESHOLD = 0.5

# 움직임 감지 설정
MIN_AREA = 2500          # 움직임으로 인정할 최소 면적
TOO_CLOSE_AREA = 90000   # 너무 가까운 물체로 판단할 면적

# 화면 구역 기준
CENTER_MARGIN = 80

# 속도 설정
FORWARD_SPEED = 0.40
TURN_SPEED = 0.32
SEARCH_SPEED = 0.25

# =========================
# GPIO 모터 핀 설정
# BCM 번호 기준
# =========================

LEFT_IN1 = 17
LEFT_IN2 = 27
LEFT_PWM = 18

RIGHT_IN1 = 22
RIGHT_IN2 = 23
RIGHT_PWM = 13

left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2)
right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2)

left_pwm = PWMOutputDevice(LEFT_PWM)
right_pwm = PWMOutputDevice(RIGHT_PWM)

# =========================
# 모터 함수
# =========================

def set_speed(speed):
    speed = max(0.0, min(1.0, speed))
    left_pwm.value = speed
    right_pwm.value = speed


def forward(speed=FORWARD_SPEED):
    set_speed(speed)
    left_motor.forward()
    right_motor.forward()


def stop():
    left_motor.stop()
    right_motor.stop()
    left_pwm.value = 0
    right_pwm.value = 0


def turn_left(speed=TURN_SPEED):
    set_speed(speed)
    left_motor.backward()
    right_motor.forward()


def turn_right(speed=TURN_SPEED):
    set_speed(speed)
    left_motor.forward()
    right_motor.backward()


def search_turn(speed=SEARCH_SPEED):
    # 사람이 안 보이면 천천히 제자리 회전
    set_speed(speed)
    left_motor.forward()
    right_motor.backward()


# =========================
# TFLite 모델 로드
# =========================

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

# =========================
# safe / danger 추론 함수
# =========================

def predict_safe_danger(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    input_data = np.expand_dims(img, axis=0)

    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32)
    else:
        input_data = input_data.astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    score = float(output[0][0])

    if score >= SAFE_THRESHOLD:
        label = "safe"
        confidence = score
    else:
        label = "danger"
        confidence = 1.0 - score

    return label, confidence


# =========================
# 움직이는 물체 감지 함수
# =========================

def detect_motion(frame, back_subtractor):
    fg_mask = back_subtractor.apply(frame)

    # 노이즈 제거
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, thresh

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_AREA:
        return None, thresh

    x, y, w, h = cv2.boundingRect(largest)
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


# =========================
# 카메라 열기
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    stop()
    exit()

# 배경 제거기
back_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=120,
    varThreshold=50,
    detectShadows=True
)

print("카메라 기반 자율 구동 시작")

# =========================
# 메인 루프
# =========================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            stop()
            break

        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2

        # 1. 움직이는 물체 감지
        motion_info, motion_mask = detect_motion(frame, back_subtractor)

        # 2. safe / danger 판별
        label, confidence = predict_safe_danger(frame)

        robot_state = "STOP"

        # =========================
        # 자율 구동 판단
        # =========================

        if motion_info is None:
            # 움직이는 물체가 없으면 탐색 회전
            search_turn(SEARCH_SPEED)
            robot_state = "SEARCH"

        else:
            cx = motion_info["cx"]
            area = motion_info["area"]

            # 너무 가까우면 정지
            if area >= TOO_CLOSE_AREA:
                stop()
                robot_state = "TOO CLOSE STOP"

            # danger면 무조건 정지
            elif label == "danger":
                stop()
                robot_state = "DANGER STOP"

            # 물체가 왼쪽에 있으면 좌회전
            elif cx < center_x - CENTER_MARGIN:
                turn_left(TURN_SPEED)
                robot_state = "TURN LEFT"

            # 물체가 오른쪽에 있으면 우회전
            elif cx > center_x + CENTER_MARGIN:
                turn_right(TURN_SPEED)
                robot_state = "TURN RIGHT"

            # 중앙에 있고 safe면 전진
            else:
                forward(FORWARD_SPEED)
                robot_state = "FORWARD"

        # =========================
        # 디버그 화면 표시
        # =========================

        if DEBUG:
            display = frame.copy()

            # 중앙 기준선 표시
            cv2.line(
                display,
                (center_x, 0),
                (center_x, frame_h),
                (255, 255, 0),
                2
            )

            cv2.line(
                display,
                (center_x - CENTER_MARGIN, 0),
                (center_x - CENTER_MARGIN, frame_h),
                (100, 100, 255),
                1
            )

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

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2
                )

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

            cv2.imshow("Robot Camera Motion Safe Drive", display)
            cv2.imshow("Motion Mask", motion_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop()
                break

except KeyboardInterrupt:
    print("강제 종료됨")
    stop()

finally:
    stop()
    cap.release()

    if DEBUG:
        cv2.destroyAllWindows()

    print("로봇 정지 완료")
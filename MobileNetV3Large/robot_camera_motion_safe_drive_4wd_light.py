# robot_camera_motion_safe_drive_4wd_light.py
# 4WD lightweight version for Raspberry Pi real-time robot driving

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from gpiozero import Motor, PWMOutputDevice, Buzzer

# ============================================================
# [1] 기본 설정 - 경량화 버전
# ============================================================

MODEL_PATH = "safe_danger_model.tflite"
CAMERA_INDEX = 0

# 모델 입력 크기는 실제 TFLite 입력 shape에서 자동으로 읽음
# 모델이 224x224로 만들어졌으면 자동으로 224 사용
DEFAULT_IMG_SIZE = 224

# 실전 로봇 구동에서는 화면 출력 OFF 권장
DEBUG = False

SAFE_THRESHOLD = 0.5

# 카메라 해상도 축소: 프레임 처리량 감소 + 지연 감소
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# AI 추론 주기: 매 프레임 추론하지 않고 N프레임마다 1번 추론
# 값이 작을수록 반응 빠름, 클수록 가벼움
AI_INTERVAL = 3

# 움직임 감지용 내부 축소 비율
# 0.5면 320x240 -> 160x120에서 움직임 계산
MOTION_SCALE = 0.5

# 움직임 감지 설정
# 축소 화면 기준으로 처리하므로 기존보다 작게 설정
MIN_AREA = 600
TOO_CLOSE_AREA = 22000
CENTER_MARGIN = 45

# ROI 여유 범위: 움직임 박스 주변을 조금 더 크게 잘라서 AI 판단
ROI_PADDING = 25

# 속도 설정
FORWARD_SPEED = 0.40
TURN_SPEED = 0.32
SEARCH_SPEED = 0.25

# ============================================================
# [2] GPIO 핀 설정 - 4륜 기준
# ============================================================

LF_IN1 = 17
LF_IN2 = 27
LR_IN1 = 5
LR_IN2 = 6
RF_IN1 = 22
RF_IN2 = 23
RR_IN1 = 24
RR_IN2 = 25

LEFT_PWM = 18
RIGHT_PWM = 13
BUZZER_PIN = 26

# ============================================================
# [3] 모터 / PWM / 부저 객체 생성
# ============================================================

left_front_motor = Motor(forward=LF_IN1, backward=LF_IN2)
left_rear_motor = Motor(forward=LR_IN1, backward=LR_IN2)
right_front_motor = Motor(forward=RF_IN1, backward=RF_IN2)
right_rear_motor = Motor(forward=RR_IN1, backward=RR_IN2)

left_pwm = PWMOutputDevice(LEFT_PWM)
right_pwm = PWMOutputDevice(RIGHT_PWM)
buzzer = Buzzer(BUZZER_PIN)

# ============================================================
# [4] 모터 제어 함수 - 같은 명령 반복 전송 방지
# ============================================================

_last_motor_state = None
_last_left_speed = None
_last_right_speed = None
_last_alarm_state = None


def clamp_speed(speed):
    return max(0.0, min(1.0, float(speed)))


def set_speed(left_speed, right_speed=None):
    global _last_left_speed, _last_right_speed

    if right_speed is None:
        right_speed = left_speed

    left_speed = clamp_speed(left_speed)
    right_speed = clamp_speed(right_speed)

    if left_speed == _last_left_speed and right_speed == _last_right_speed:
        return

    left_pwm.value = left_speed
    right_pwm.value = right_speed
    _last_left_speed = left_speed
    _last_right_speed = right_speed


def left_group_forward():
    left_front_motor.forward()
    left_rear_motor.forward()


def left_group_backward():
    left_front_motor.backward()
    left_rear_motor.backward()


def right_group_forward():
    right_front_motor.forward()
    right_rear_motor.forward()


def right_group_backward():
    right_front_motor.backward()
    right_rear_motor.backward()


def apply_motor_state(state, speed=0.0):
    """
    state: stop / forward / left / right / search
    같은 상태가 반복되면 GPIO 명령을 다시 보내지 않아 CPU 사용과 떨림을 줄임.
    """
    global _last_motor_state

    if state == _last_motor_state:
        set_speed(speed)
        return

    if state == "stop":
        left_front_motor.stop()
        left_rear_motor.stop()
        right_front_motor.stop()
        right_rear_motor.stop()
        set_speed(0.0)

    elif state == "forward":
        set_speed(speed)
        left_group_forward()
        right_group_forward()

    elif state == "left":
        set_speed(speed)
        left_group_backward()
        right_group_forward()

    elif state == "right":
        set_speed(speed)
        left_group_forward()
        right_group_backward()

    elif state == "search":
        set_speed(speed)
        left_group_forward()
        right_group_backward()

    _last_motor_state = state


def forward(speed=FORWARD_SPEED):
    apply_motor_state("forward", speed)


def stop():
    apply_motor_state("stop", 0.0)


def turn_left(speed=TURN_SPEED):
    apply_motor_state("left", speed)


def turn_right(speed=TURN_SPEED):
    apply_motor_state("right", speed)


def search_turn(speed=SEARCH_SPEED):
    apply_motor_state("search", speed)


def set_alarm(on):
    global _last_alarm_state

    on = bool(on)
    if on == _last_alarm_state:
        return

    if on:
        buzzer.on()
    else:
        buzzer.off()

    _last_alarm_state = on


def alarm_on():
    set_alarm(True)


def alarm_off():
    set_alarm(False)

# ============================================================
# [5] TFLite 모델 로드
# ============================================================

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]

# 입력 크기 자동 확인: 보통 [1, 224, 224, 3]
input_shape = input_details[0].get("shape", None)
if input_shape is not None and len(input_shape) >= 3:
    IMG_SIZE = int(input_shape[1])
else:
    IMG_SIZE = DEFAULT_IMG_SIZE

print("Input details:", input_details)
print("Output details:", output_details)
print("Lightweight 4WD mode: DEBUG=False, camera=320x240, ROI inference, AI_INTERVAL=", AI_INTERVAL)

# ============================================================
# [6] safe / danger 판별 함수 - ROI + dtype 보정
# ============================================================


def get_score_from_output(output):
    """
    float32 / int8 / uint8 출력 모두 대응.
    """
    value = float(output.reshape(-1)[0])

    if output_dtype in (np.uint8, np.int8):
        quant = output_details[0].get("quantization", (0.0, 0))
        scale, zero_point = quant
        if scale and scale > 0:
            value = (value - zero_point) * scale

    # sigmoid 단일 출력 기준. 값이 범위를 살짝 벗어나면 보정.
    return max(0.0, min(1.0, value))


def predict_safe_danger(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(img, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32)
    else:
        input_data = input_data.astype(input_dtype)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)
    score = get_score_from_output(output)

    if score >= SAFE_THRESHOLD:
        return "safe", score
    return "danger", 1.0 - score

# ============================================================
# [7] 움직임 감지 함수 - 축소 프레임에서 계산
# ============================================================


def detect_motion(small_frame, back_subtractor):
    fg_mask = back_subtractor.apply(small_frame)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
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


def crop_roi_from_motion(frame, motion_info):
    """
    축소 프레임에서 찾은 움직임 좌표를 원본 프레임 좌표로 환산해서 ROI만 자름.
    """
    scale = 1.0 / MOTION_SCALE
    frame_h, frame_w = frame.shape[:2]

    x = int(motion_info["x"] * scale)
    y = int(motion_info["y"] * scale)
    w = int(motion_info["w"] * scale)
    h = int(motion_info["h"] * scale)

    x1 = max(0, x - ROI_PADDING)
    y1 = max(0, y - ROI_PADDING)
    x2 = min(frame_w, x + w + ROI_PADDING)
    y2 = min(frame_h, y + h + ROI_PADDING)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame

    return roi

# ============================================================
# [8] 카메라 열기 및 설정
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

# 카메라 지연 감소 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    stop()
    alarm_off()
    raise SystemExit

back_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=80,
    varThreshold=50,
    detectShadows=False
)

print("4륜 경량화 자율 구동 시작")

# ============================================================
# [9] 메인 루프
# ============================================================

frame_count = 0
last_label = "safe"
last_confidence = 0.0
robot_state = "STOP"

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            stop()
            alarm_off()
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2

        # 움직임 감지는 축소 프레임에서 수행
        small_frame = cv2.resize(
            frame,
            (int(frame_w * MOTION_SCALE), int(frame_h * MOTION_SCALE)),
            interpolation=cv2.INTER_AREA
        )
        motion_info, motion_mask = detect_motion(small_frame, back_subtractor)

        if motion_info is None:
            last_label = "safe"
            last_confidence = 0.0
            alarm_off()
            search_turn(SEARCH_SPEED)
            robot_state = "SEARCH"

        else:
            cx = int(motion_info["cx"] / MOTION_SCALE)
            area = motion_info["area"]

            # 너무 가까우면 AI 추론보다 정지를 우선함
            if area >= TOO_CLOSE_AREA:
                alarm_off()
                stop()
                robot_state = "TOO CLOSE STOP"

            else:
                # 매 프레임 전체 추론 대신, 일정 주기마다 움직임 ROI만 추론
                if frame_count % AI_INTERVAL == 0:
                    roi = crop_roi_from_motion(frame, motion_info)
                    last_label, last_confidence = predict_safe_danger(roi)

                if last_label == "danger":
                    stop()
                    alarm_on()
                    robot_state = "DANGER STOP + BUZZER"

                elif cx < center_x - CENTER_MARGIN:
                    alarm_off()
                    turn_left(TURN_SPEED)
                    robot_state = "TURN LEFT"

                elif cx > center_x + CENTER_MARGIN:
                    alarm_off()
                    turn_right(TURN_SPEED)
                    robot_state = "TURN RIGHT"

                else:
                    alarm_off()
                    forward(FORWARD_SPEED)
                    robot_state = "FORWARD"

        if DEBUG:
            display = frame.copy()

            cv2.line(display, (center_x, 0), (center_x, frame_h), (255, 255, 0), 2)
            cv2.line(display, (center_x - CENTER_MARGIN, 0), (center_x - CENTER_MARGIN, frame_h), (100, 100, 255), 1)
            cv2.line(display, (center_x + CENTER_MARGIN, 0), (center_x + CENTER_MARGIN, frame_h), (100, 100, 255), 1)

            if motion_info is not None:
                scale = 1.0 / MOTION_SCALE
                x = int(motion_info["x"] * scale)
                y = int(motion_info["y"] * scale)
                w = int(motion_info["w"] * scale)
                h = int(motion_info["h"] * scale)
                cx = int(motion_info["cx"] * scale)
                cy = int(motion_info["cy"] * scale)
                area = motion_info["area"]

                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(display, f"Motion Area: {int(area)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            text1 = f"AI: {last_label.upper()} {last_confidence * 100:.1f}%"
            text2 = f"ROBOT: {robot_state}"
            cv2.putText(display, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if last_label == "safe" else (0, 0, 255), 2)
            cv2.putText(display, text2, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("4WD Robot Lightweight Drive", display)
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

    print("4륜 경량화 로봇 정지 완료")

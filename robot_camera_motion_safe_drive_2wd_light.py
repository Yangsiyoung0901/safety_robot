# robot_camera_motion_safe_drive_2wd_light.py
# 라즈베리파이 실시간 구동용 경량화 버전

import cv2
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter
from gpiozero import Motor, PWMOutputDevice

# =========================
# 기본 설정
# =========================

MODEL_PATH = "safe_danger_model.tflite"
CAMERA_INDEX = 0

# 로봇 실제 구동 시에는 반드시 False 권장
# True이면 화면 출력 때문에 FPS가 떨어짐
DEBUG = False

SAFE_THRESHOLD = 0.5

# 카메라 입력 해상도 축소: 처리량 감소
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# AI 추론 주기
# 1이면 매 프레임 추론, 3이면 3프레임마다 1번 추론
# 라즈베리파이에서는 3~5 권장
AI_INTERVAL = 3

# 움직임 감지용 축소 비율
# 0.5이면 320x240 프레임을 160x120으로 줄여서 움직임 감지
MOTION_SCALE = 0.5

# 움직임 감지 설정
# MOTION_SCALE을 적용한 작은 화면 기준 면적
MIN_AREA = 600
TOO_CLOSE_AREA = 23000

# ROI 여유 영역
ROI_MARGIN = 30

# 화면 구역 기준
CENTER_MARGIN = 50

# 속도 설정
FORWARD_SPEED = 0.40
TURN_SPEED = 0.32
SEARCH_SPEED = 0.25

# 모터 명령 반복 전송 방지
# 같은 명령을 매 프레임 계속 보내지 않도록 함
last_motor_state = None

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
    speed = max(0.0, min(1.0, float(speed)))
    left_pwm.value = speed
    right_pwm.value = speed


def apply_motor_state(state, speed=0.0):
    """같은 모터 명령을 반복해서 보내지 않기 위한 통합 함수."""
    global last_motor_state

    command = (state, round(float(speed), 2))
    if command == last_motor_state:
        return

    last_motor_state = command

    if state == "FORWARD":
        set_speed(speed)
        left_motor.forward()
        right_motor.forward()

    elif state == "TURN_LEFT":
        set_speed(speed)
        left_motor.backward()
        right_motor.forward()

    elif state == "TURN_RIGHT":
        set_speed(speed)
        left_motor.forward()
        right_motor.backward()

    elif state == "SEARCH":
        set_speed(speed)
        left_motor.forward()
        right_motor.backward()

    else:
        left_motor.stop()
        right_motor.stop()
        left_pwm.value = 0
        right_pwm.value = 0


def stop():
    apply_motor_state("STOP", 0.0)


# =========================
# TFLite 모델 로드
# =========================

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_info = input_details[0]
output_info = output_details[0]

input_index = input_info["index"]
output_index = output_info["index"]
input_dtype = input_info["dtype"]

# 모델이 요구하는 입력 크기를 자동으로 읽음
# 보통 [1, 224, 224, 3] 또는 [1, 160, 160, 3]
input_shape = input_info["shape"]
MODEL_H = int(input_shape[1])
MODEL_W = int(input_shape[2])

print("Input details:", input_details)
print("Output details:", output_details)
print(f"Model input size: {MODEL_W}x{MODEL_H}, dtype: {input_dtype}")

# =========================
# safe / danger 추론 함수
# =========================

def dequantize_output_if_needed(output):
    """INT8/UINT8 양자화 모델 출력이면 실제 float score로 복원."""
    quant = output_info.get("quantization", (0.0, 0))
    scale, zero_point = quant

    if scale and scale > 0:
        return (output.astype(np.float32) - zero_point) * scale

    return output


def predict_safe_danger(frame):
    """프레임 또는 ROI 이미지를 받아 safe/danger 추론."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)

    input_data = np.expand_dims(img, axis=0)

    if input_dtype == np.float32:
        # 기존 코드와 호환성을 위해 0~255 float 입력 유지
        # 모델 학습 때 Rescaling(1./255)을 모델 안에 넣었다면 이 방식이 맞음
        input_data = input_data.astype(np.float32)
    else:
        # INT8/UINT8 양자화 모델 대응
        input_data = input_data.astype(input_dtype)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)
    output = dequantize_output_if_needed(output)

    score = float(output.reshape(-1)[0])

    if score >= SAFE_THRESHOLD:
        return "safe", score

    return "danger", 1.0 - score


# =========================
# 움직이는 물체 감지 함수
# =========================

def detect_motion(frame, back_subtractor):
    """작은 프레임에서 움직임을 감지하고 원본 좌표로 복원."""
    small = cv2.resize(
        frame,
        None,
        fx=MOTION_SCALE,
        fy=MOTION_SCALE,
        interpolation=cv2.INTER_AREA
    )

    fg_mask = back_subtractor.apply(small)

    # 그림자 값 제거: detectShadows=False라 대부분 불필요하지만 안전용으로 유지
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # 작은 화면 기준이므로 커널도 작게 사용
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, thresh

    largest = max(contours, key=cv2.contourArea)
    area_small = cv2.contourArea(largest)

    if area_small < MIN_AREA:
        return None, thresh

    x, y, w, h = cv2.boundingRect(largest)

    inv_scale = 1.0 / MOTION_SCALE
    x = int(x * inv_scale)
    y = int(y * inv_scale)
    w = int(w * inv_scale)
    h = int(h * inv_scale)
    area = float(area_small * inv_scale * inv_scale)

    cx = x + w // 2
    cy = y + h // 2

    motion_info = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "area": area,
    }

    return motion_info, thresh


# =========================
# ROI 추출 함수
# =========================

def crop_motion_roi(frame, motion_info):
    """움직이는 물체 주변만 잘라서 AI 추론량을 줄임."""
    frame_h, frame_w = frame.shape[:2]

    x = motion_info["x"]
    y = motion_info["y"]
    w = motion_info["w"]
    h = motion_info["h"]

    x1 = max(0, x - ROI_MARGIN)
    y1 = max(0, y - ROI_MARGIN)
    x2 = min(frame_w, x + w + ROI_MARGIN)
    y2 = min(frame_h, y + h + ROI_MARGIN)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    return roi


# =========================
# 카메라 열기
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)

# 카메라 버퍼 최소화: 지연 감소
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    stop()
    raise SystemExit

# 배경 제거기
# detectShadows=False로 그림자 계산 제거 → 더 빠름
back_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=60,
    varThreshold=50,
    detectShadows=False
)

print("경량화 로봇 자율 구동 시작")

# =========================
# 메인 루프
# =========================

frame_count = 0
last_label = "danger"
last_confidence = 1.0
last_ai_time = 0.0
prev_log_time = 0.0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            stop()
            break

        frame_count += 1

        frame_h, frame_w = frame.shape[:2]
        center_x = frame_w // 2

        # 1. 움직이는 물체 감지
        motion_info, motion_mask = detect_motion(frame, back_subtractor)

        # 2. AI 추론
        # 움직임이 있을 때만, 그리고 AI_INTERVAL마다 추론
        if motion_info is not None and frame_count % AI_INTERVAL == 0:
            roi = crop_motion_roi(frame, motion_info)
            last_ai_time = time.time()
            last_label, last_confidence = predict_safe_danger(roi)

        label = last_label
        confidence = last_confidence

        # 3. 자율 구동 판단
        if motion_info is None:
            apply_motor_state("SEARCH", SEARCH_SPEED)
            robot_state = "SEARCH"

        else:
            cx = motion_info["cx"]
            area = motion_info["area"]

            if area >= TOO_CLOSE_AREA:
                apply_motor_state("STOP", 0.0)
                robot_state = "TOO CLOSE STOP"

            elif label == "danger":
                apply_motor_state("STOP", 0.0)
                robot_state = "DANGER STOP"

            elif cx < center_x - CENTER_MARGIN:
                apply_motor_state("TURN_LEFT", TURN_SPEED)
                robot_state = "TURN LEFT"

            elif cx > center_x + CENTER_MARGIN:
                apply_motor_state("TURN_RIGHT", TURN_SPEED)
                robot_state = "TURN RIGHT"

            else:
                apply_motor_state("FORWARD", FORWARD_SPEED)
                robot_state = "FORWARD"

        # 터미널 로그도 너무 자주 찍으면 느려짐
        now = time.time()
        if now - prev_log_time >= 1.0:
            prev_log_time = now
            print(
                f"STATE={robot_state} | AI={label} {confidence * 100:.1f}% | "
                f"motion={'yes' if motion_info else 'no'}"
            )

        # 4. 디버그 화면 표시
        if DEBUG:
            display = frame.copy()

            cv2.line(display, (center_x, 0), (center_x, frame_h), (255, 255, 0), 2)
            cv2.line(display, (center_x - CENTER_MARGIN, 0), (center_x - CENTER_MARGIN, frame_h), (100, 100, 255), 1)
            cv2.line(display, (center_x + CENTER_MARGIN, 0), (center_x + CENTER_MARGIN, frame_h), (100, 100, 255), 1)

            if motion_info is not None:
                x = motion_info["x"]
                y = motion_info["y"]
                w = motion_info["w"]
                h = motion_info["h"]
                cx = motion_info["cx"]
                cy = motion_info["cy"]
                area = motion_info["area"]

                cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(display, f"Area: {int(area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.putText(display, f"AI: {label.upper()} {confidence * 100:.1f}%", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if label == "safe" else (0, 0, 255), 2)
            cv2.putText(display, f"ROBOT: {robot_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Robot Camera Motion Safe Drive Light", display)
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

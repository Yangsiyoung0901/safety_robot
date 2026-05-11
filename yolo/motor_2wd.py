from gpiozero import Motor, PWMOutputDevice  # 2륜 모터와 PWM 제어를 위해 gpiozero 클래스를 가져옵니다.

from config import FORWARD_SPEED, SEARCH_SPEED, TURN_SPEED  # 기본 속도 설정값을 가져옵니다.

LEFT_IN1 = 17  # 왼쪽 모터 정방향 입력 핀입니다.
LEFT_IN2 = 27  # 왼쪽 모터 역방향 입력 핀입니다.
LEFT_PWM = 18  # 왼쪽 모터 속도 제어 PWM 핀입니다.
RIGHT_IN1 = 22  # 오른쪽 모터 정방향 입력 핀입니다.
RIGHT_IN2 = 23  # 오른쪽 모터 역방향 입력 핀입니다.
RIGHT_PWM = 13  # 오른쪽 모터 속도 제어 PWM 핀입니다.


class Motor2WD:  # 2륜 로봇 모터 제어 클래스입니다.
    def __init__(self) -> None:  # 객체가 생성될 때 한 번 실행되는 초기화 함수입니다.
        self.left_motor = Motor(forward=LEFT_IN1, backward=LEFT_IN2)  # 왼쪽 모터 객체를 만듭니다.
        self.right_motor = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2)  # 오른쪽 모터 객체를 만듭니다.
        self.left_pwm = PWMOutputDevice(LEFT_PWM)  # 왼쪽 PWM 출력 객체를 만듭니다.
        self.right_pwm = PWMOutputDevice(RIGHT_PWM)  # 오른쪽 PWM 출력 객체를 만듭니다.
        self.last_command = None  # 마지막 명령을 저장해서 반복 명령을 줄입니다.

    def set_speed(self, speed: float) -> None:  # 양쪽 모터 속도를 같은 값으로 설정하는 함수입니다.
        speed = max(0.0, min(1.0, float(speed)))  # 속도를 0.0부터 1.0 사이로 제한합니다.
        self.left_pwm.value = speed  # 왼쪽 PWM 값을 설정합니다.
        self.right_pwm.value = speed  # 오른쪽 PWM 값을 설정합니다.

    def apply(self, command: str) -> None:  # 주행 명령 문자열을 실제 모터 동작으로 바꾸는 함수입니다.
        if command == self.last_command:  # 이전 명령과 같으면 실행합니다.
            return  # 같은 GPIO 명령 반복을 피하기 위해 아무것도 하지 않습니다.
        self.last_command = command  # 현재 명령을 마지막 명령으로 저장합니다.
        if command == "FORWARD":  # 전진 명령이면 실행합니다.
            self.set_speed(FORWARD_SPEED)  # 전진 속도로 PWM을 설정합니다.
            self.left_motor.forward()  # 왼쪽 모터를 정방향으로 돌립니다.
            self.right_motor.forward()  # 오른쪽 모터를 정방향으로 돌립니다.
        elif command == "TURN_LEFT":  # 좌회전 명령이면 실행합니다.
            self.set_speed(TURN_SPEED)  # 회전 속도로 PWM을 설정합니다.
            self.left_motor.backward()  # 왼쪽 모터를 역방향으로 돌립니다.
            self.right_motor.forward()  # 오른쪽 모터를 정방향으로 돌립니다.
        elif command == "TURN_RIGHT":  # 우회전 명령이면 실행합니다.
            self.set_speed(TURN_SPEED)  # 회전 속도로 PWM을 설정합니다.
            self.left_motor.forward()  # 왼쪽 모터를 정방향으로 돌립니다.
            self.right_motor.backward()  # 오른쪽 모터를 역방향으로 돌립니다.
        elif command == "SEARCH":  # 탐색 회전 명령이면 실행합니다.
            self.set_speed(SEARCH_SPEED)  # 탐색 속도로 PWM을 설정합니다.
            self.left_motor.forward()  # 왼쪽 모터를 정방향으로 돌립니다.
            self.right_motor.backward()  # 오른쪽 모터를 역방향으로 돌립니다.
        else:  # STOP 또는 알 수 없는 명령이면 실행합니다.
            self.stop()  # 로봇을 정지합니다.

    def stop(self) -> None:  # 로봇을 정지시키는 함수입니다.
        self.left_motor.stop()  # 왼쪽 모터를 정지합니다.
        self.right_motor.stop()  # 오른쪽 모터를 정지합니다.
        self.left_pwm.value = 0.0  # 왼쪽 PWM을 0으로 설정합니다.
        self.right_pwm.value = 0.0  # 오른쪽 PWM을 0으로 설정합니다.
        self.last_command = "STOP"  # 마지막 명령을 STOP으로 저장합니다.

    def cleanup(self) -> None:  # 프로그램 종료 시 호출할 정리 함수입니다.
        self.stop()  # 종료 전 로봇을 안전하게 정지합니다.

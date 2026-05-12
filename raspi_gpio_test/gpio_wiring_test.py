#!/usr/bin/env python3  # 이 파일을 리눅스에서 직접 실행할 때 Python 3로 실행하라는 뜻입니다.
"""  # 여러 줄 설명문 시작입니다.
Raspberry Pi 5 wiring test for motor driver, buzzer, and ultrasonic sensor.  # 이 파일의 목적 설명입니다.
  # 빈 줄도 설명문 안에서는 보기 좋게 구분하는 역할을 합니다.
Final wiring:  # 아래는 최종 배선표입니다.
- Physical pin 29 / GPIO5  : left motor driver IN1  # 왼쪽 모터 방향 제어 1번 선입니다.
- Physical pin 31 / GPIO6  : left motor driver IN2  # 왼쪽 모터 방향 제어 2번 선입니다.
- Physical pin 32 / GPIO12 : right motor driver IN1  # 오른쪽 모터 방향 제어 1번 선입니다.
- Physical pin 33 / GPIO13 : right motor driver IN2  # 오른쪽 모터 방향 제어 2번 선입니다.
- Physical pin 35 / GPIO19 : buzzer +  # 부저의 플러스 선입니다.
- Physical pin 36 / GPIO16 : ultrasonic Trig  # 초음파 센서에 측정 신호를 보내는 선입니다.
- Physical pin 37 / GPIO26 : ultrasonic Echo  # 초음파 센서에서 되돌아오는 신호를 받는 선입니다.
  # 빈 줄도 설명문 안에서는 보기 좋게 구분하는 역할을 합니다.
Warning:  # 중요한 주의사항입니다.
Many HC-SR04 ultrasonic modules output 5V on Echo.  # HC-SR04 Echo 핀은 보통 5V 신호가 나옵니다.
Raspberry Pi GPIO pins accept 3.3V only, so use a voltage divider  # 라즈베리파이 GPIO는 3.3V까지만 안전합니다.
or level shifter between Echo and GPIO26.  # Echo와 GPIO26 사이에는 전압분배 회로나 레벨시프터를 넣으세요.
"""  # 여러 줄 설명문 끝입니다.

from __future__ import annotations  # 타입 힌트 처리를 최신 방식으로 하게 해주는 설정입니다.

from time import sleep  # sleep은 몇 초 동안 잠깐 멈출 때 사용하는 함수입니다.

from gpiozero import Buzzer, DistanceSensor, PWMOutputDevice  # GPIO 장치를 쉽게 제어하는 클래스들을 가져옵니다.
from gpiozero.exc import DistanceSensorNoEcho  # 초음파 센서가 반사 신호를 못 받았을 때의 예외를 가져옵니다.


LEFT_IN1 = 5  # 왼쪽 모터 드라이버 IN1에 연결된 BCM GPIO 번호입니다. 물리핀 29번입니다.
LEFT_IN2 = 6  # 왼쪽 모터 드라이버 IN2에 연결된 BCM GPIO 번호입니다. 물리핀 31번입니다.
RIGHT_IN1 = 12  # 오른쪽 모터 드라이버 IN1에 연결된 BCM GPIO 번호입니다. 물리핀 32번입니다.
RIGHT_IN2 = 13  # 오른쪽 모터 드라이버 IN2에 연결된 BCM GPIO 번호입니다. 물리핀 33번입니다.
BUZZER_PIN = 19  # 부저 +에 연결된 BCM GPIO 번호입니다. 물리핀 35번입니다.
ULTRASONIC_TRIG = 16  # 초음파 센서 Trig에 연결된 BCM GPIO 번호입니다. 물리핀 36번입니다.
ULTRASONIC_ECHO = 26  # 초음파 센서 Echo에 연결된 BCM GPIO 번호입니다. 물리핀 37번입니다.

MOTOR_TEST_SECONDS = 1.0  # 모터를 한 방향으로 테스트할 때 몇 초 동안 돌릴지 정합니다.
DEFAULT_MOTOR_SPEED = 0.6  # 속도제어 테스트에서 기본으로 사용할 모터 속도입니다. 0.0은 정지, 1.0은 최대입니다.
PWM_FREQUENCY = 1000  # 모터 속도제어에 사용할 PWM 주파수입니다. 보통 500~2000 사이에서 테스트하면 됩니다.


class MotorTester:  # 모터 드라이버 테스트 기능을 모아둔 클래스입니다.
    def __init__(self) -> None:  # MotorTester가 만들어질 때 GPIO 출력 핀을 준비합니다.
        self.left_in1 = PWMOutputDevice(LEFT_IN1, frequency=PWM_FREQUENCY)  # 왼쪽 모터 IN1 핀을 PWM 출력 장치로 설정합니다.
        self.left_in2 = PWMOutputDevice(LEFT_IN2, frequency=PWM_FREQUENCY)  # 왼쪽 모터 IN2 핀을 PWM 출력 장치로 설정합니다.
        self.right_in1 = PWMOutputDevice(RIGHT_IN1, frequency=PWM_FREQUENCY)  # 오른쪽 모터 IN1 핀을 PWM 출력 장치로 설정합니다.
        self.right_in2 = PWMOutputDevice(RIGHT_IN2, frequency=PWM_FREQUENCY)  # 오른쪽 모터 IN2 핀을 PWM 출력 장치로 설정합니다.

    def clamp_speed(self, speed: float) -> float:  # 입력된 속도값을 안전한 범위로 제한하는 함수입니다.
        return max(0.0, min(1.0, speed))  # 0.0보다 작으면 0.0, 1.0보다 크면 1.0으로 고정합니다.

    def stop(self) -> None:  # 양쪽 모터를 모두 멈추는 함수입니다.
        self.left_in1.off()  # 왼쪽 모터 IN1을 꺼서 신호를 0으로 만듭니다.
        self.left_in2.off()  # 왼쪽 모터 IN2를 꺼서 신호를 0으로 만듭니다.
        self.right_in1.off()  # 오른쪽 모터 IN1을 꺼서 신호를 0으로 만듭니다.
        self.right_in2.off()  # 오른쪽 모터 IN2를 꺼서 신호를 0으로 만듭니다.

    def left_forward(self, speed: float = 1.0) -> None:  # 왼쪽 모터를 정방향으로 돌리는 함수입니다.
        speed = self.clamp_speed(speed)  # 속도값을 0.0~1.0 사이로 제한합니다.
        self.left_in1.value = speed  # 왼쪽 IN1에 PWM 속도값을 출력합니다.
        self.left_in2.value = 0.0  # 왼쪽 IN2는 0으로 만들어 정방향이 되게 합니다.

    def left_backward(self, speed: float = 1.0) -> None:  # 왼쪽 모터를 역방향으로 돌리는 함수입니다.
        speed = self.clamp_speed(speed)  # 속도값을 0.0~1.0 사이로 제한합니다.
        self.left_in1.value = 0.0  # 왼쪽 IN1은 0으로 만듭니다.
        self.left_in2.value = speed  # 왼쪽 IN2에 PWM 속도값을 출력해서 역방향이 되게 합니다.

    def right_forward(self, speed: float = 1.0) -> None:  # 오른쪽 모터를 정방향으로 돌리는 함수입니다.
        speed = self.clamp_speed(speed)  # 속도값을 0.0~1.0 사이로 제한합니다.
        self.right_in1.value = speed  # 오른쪽 IN1에 PWM 속도값을 출력합니다.
        self.right_in2.value = 0.0  # 오른쪽 IN2는 0으로 만들어 정방향이 되게 합니다.

    def right_backward(self, speed: float = 1.0) -> None:  # 오른쪽 모터를 역방향으로 돌리는 함수입니다.
        speed = self.clamp_speed(speed)  # 속도값을 0.0~1.0 사이로 제한합니다.
        self.right_in1.value = 0.0  # 오른쪽 IN1은 0으로 만듭니다.
        self.right_in2.value = speed  # 오른쪽 IN2에 PWM 속도값을 출력해서 역방향이 되게 합니다.

    def forward(self, speed: float = 1.0) -> None:  # 양쪽 모터를 정방향으로 돌려 전진시키는 함수입니다.
        self.left_forward(speed)  # 왼쪽 모터를 지정한 속도로 정방향으로 돌립니다.
        self.right_forward(speed)  # 오른쪽 모터를 지정한 속도로 정방향으로 돌립니다.

    def backward(self, speed: float = 1.0) -> None:  # 양쪽 모터를 역방향으로 돌려 후진시키는 함수입니다.
        self.left_backward(speed)  # 왼쪽 모터를 지정한 속도로 역방향으로 돌립니다.
        self.right_backward(speed)  # 오른쪽 모터를 지정한 속도로 역방향으로 돌립니다.

    def turn_left(self, speed: float = 1.0) -> None:  # 제자리에서 왼쪽으로 도는 테스트 함수입니다.
        self.left_backward(speed)  # 왼쪽 모터는 지정한 속도로 뒤로 돌립니다.
        self.right_forward(speed)  # 오른쪽 모터는 지정한 속도로 앞으로 돌립니다.

    def turn_right(self, speed: float = 1.0) -> None:  # 제자리에서 오른쪽으로 도는 테스트 함수입니다.
        self.left_forward(speed)  # 왼쪽 모터는 지정한 속도로 앞으로 돌립니다.
        self.right_backward(speed)  # 오른쪽 모터는 지정한 속도로 뒤로 돌립니다.

    def close(self) -> None:  # 프로그램이 끝날 때 GPIO 자원을 정리하는 함수입니다.
        self.stop()  # 정리하기 전에 모터를 먼저 멈춥니다.
        self.left_in1.close()  # 왼쪽 IN1 GPIO 장치를 닫습니다.
        self.left_in2.close()  # 왼쪽 IN2 GPIO 장치를 닫습니다.
        self.right_in1.close()  # 오른쪽 IN1 GPIO 장치를 닫습니다.
        self.right_in2.close()  # 오른쪽 IN2 GPIO 장치를 닫습니다.


def run_motor_step(name: str, action, motors: MotorTester, speed: float = 1.0) -> None:  # 모터 동작 하나를 짧게 실행하는 함수입니다.
    print(f"\n[MOTOR] {name} - speed {speed:.2f}, {MOTOR_TEST_SECONDS:.1f}s")  # 어떤 모터 테스트를 하는지 화면에 출력합니다.
    action(speed)  # 전달받은 모터 동작 함수를 지정한 속도로 실행합니다.
    sleep(MOTOR_TEST_SECONDS)  # 정해진 시간만큼 모터를 돌립니다.
    motors.stop()  # 테스트 시간이 지나면 모터를 멈춥니다.
    sleep(0.3)  # 다음 테스트 전에 0.3초 쉬어서 동작을 구분합니다.


def test_motors(motors: MotorTester) -> None:  # 전체 모터 테스트를 순서대로 실행하는 함수입니다.
    print("\nMotor test will start.")  # 모터 테스트 시작 안내를 출력합니다.
    print("Lift the wheels off the ground if possible.")  # 바퀴를 띄우라는 안전 안내를 출력합니다.
    input("Press Enter to continue...")  # 사용자가 Enter를 누를 때까지 기다립니다.

    run_motor_step("left motor forward", motors.left_forward, motors)  # 왼쪽 모터 정방향을 테스트합니다.
    run_motor_step("left motor backward", motors.left_backward, motors)  # 왼쪽 모터 역방향을 테스트합니다.
    run_motor_step("right motor forward", motors.right_forward, motors)  # 오른쪽 모터 정방향을 테스트합니다.
    run_motor_step("right motor backward", motors.right_backward, motors)  # 오른쪽 모터 역방향을 테스트합니다.
    run_motor_step("both motors forward", motors.forward, motors)  # 양쪽 모터 전진을 테스트합니다.
    run_motor_step("both motors backward", motors.backward, motors)  # 양쪽 모터 후진을 테스트합니다.
    run_motor_step("turn left", motors.turn_left, motors)  # 좌회전 동작을 테스트합니다.
    run_motor_step("turn right", motors.turn_right, motors)  # 우회전 동작을 테스트합니다.

    print("\nMotor test complete.")  # 모터 테스트 완료 안내를 출력합니다.


def ask_motor_speed() -> float:  # 사용자에게 모터 속도를 입력받는 함수입니다.
    text = input(f"Motor speed 0.0~1.0, Enter={DEFAULT_MOTOR_SPEED}: ").strip()  # 속도값을 입력받습니다.
    if text == "":  # 아무것도 입력하지 않고 Enter만 누른 경우입니다.
        return DEFAULT_MOTOR_SPEED  # 기본 속도값을 사용합니다.
    try:  # 숫자로 바꾸는 중 오류가 날 수 있어서 예외 처리를 시작합니다.
        speed = float(text)  # 입력 문자열을 실수 숫자로 바꿉니다.
    except ValueError:  # 숫자로 바꿀 수 없는 값을 입력했을 때 실행됩니다.
        print(f"Invalid speed. Using default speed {DEFAULT_MOTOR_SPEED}.")  # 잘못된 입력이라 기본값을 쓴다고 알려줍니다.
        return DEFAULT_MOTOR_SPEED  # 기본 속도값을 사용합니다.
    return max(0.0, min(1.0, speed))  # 입력된 속도를 0.0~1.0 사이로 제한해서 돌려줍니다.


def test_motors_with_speed(motors: MotorTester) -> None:  # 사용자가 정한 속도로 모터를 테스트하는 함수입니다.
    print("\nMotor speed control test will start.")  # 속도제어 테스트 시작 안내를 출력합니다.
    print("This uses PWM on the IN pins because ENA/ENB pins are not in the current wiring.")  # 현재 배선 기준 속도제어 방식 설명입니다.
    print("Lift the wheels off the ground if possible.")  # 바퀴를 띄우라는 안전 안내를 출력합니다.
    speed = ask_motor_speed()  # 사용자에게 속도를 입력받습니다.
    input("Press Enter to continue...")  # 사용자가 Enter를 누를 때까지 기다립니다.

    run_motor_step("both motors forward", motors.forward, motors, speed)  # 지정 속도로 전진을 테스트합니다.
    run_motor_step("both motors backward", motors.backward, motors, speed)  # 지정 속도로 후진을 테스트합니다.
    run_motor_step("turn left", motors.turn_left, motors, speed)  # 지정 속도로 좌회전을 테스트합니다.
    run_motor_step("turn right", motors.turn_right, motors, speed)  # 지정 속도로 우회전을 테스트합니다.

    print("\nMotor speed control test complete.")  # 속도제어 테스트 완료 안내를 출력합니다.


def test_buzzer(buzzer: Buzzer) -> None:  # 부저를 테스트하는 함수입니다.
    print("\n[BUZZER] Beeping 3 times.")  # 부저 테스트 시작 안내를 출력합니다.
    for _ in range(3):  # 총 3번 반복해서 삐 소리를 냅니다.
        buzzer.on()  # 부저를 켭니다.
        sleep(0.2)  # 0.2초 동안 켜둡니다.
        buzzer.off()  # 부저를 끕니다.
        sleep(0.2)  # 0.2초 동안 쉬었다가 다음 삐 소리를 냅니다.
    print("Buzzer test complete.")  # 부저 테스트 완료 안내를 출력합니다.


def test_ultrasonic(sensor: DistanceSensor) -> None:  # 초음파 센서 거리를 테스트하는 함수입니다.
    print("\n[ULTRASONIC] Measuring distance 10 times.")  # 초음파 테스트 시작 안내를 출력합니다.
    print("Place an object about 10-100 cm in front of the sensor.")  # 물체를 센서 앞에 두라는 안내를 출력합니다.
    for index in range(1, 11):  # 1부터 10까지 총 10번 측정합니다.
        try:  # 거리 측정 중 오류가 날 수 있어서 예외 처리를 시작합니다.
            distance_cm = sensor.distance * 100  # gpiozero는 미터 단위라서 100을 곱해 cm로 바꿉니다.
            print(f"{index:02d}: {distance_cm:6.1f} cm")  # 측정 번호와 거리를 보기 좋게 출력합니다.
        except DistanceSensorNoEcho:  # 반사 신호가 없어서 측정하지 못했을 때 실행됩니다.
            print(f"{index:02d}: no echo")  # Echo 신호를 못 받았다고 출력합니다.
        sleep(0.5)  # 다음 측정 전 0.5초 기다립니다.
    print("Ultrasonic test complete.")  # 초음파 테스트 완료 안내를 출력합니다.


def print_menu() -> None:  # 사용자에게 선택 메뉴를 보여주는 함수입니다.
    print("\n=== Raspberry Pi 5 GPIO wiring test ===")  # 메뉴 제목을 출력합니다.
    print("1. Test buzzer")  # 1번 메뉴는 부저 테스트입니다.
    print("2. Test ultrasonic sensor")  # 2번 메뉴는 초음파 센서 테스트입니다.
    print("3. Test motors")  # 3번 메뉴는 모터 테스트입니다.
    print("4. Test motors with speed control")  # 4번 메뉴는 속도제어 모터 테스트입니다.
    print("5. Test all")  # 5번 메뉴는 전체 테스트입니다.
    print("q. Quit")  # q를 입력하면 프로그램을 종료합니다.


def main() -> None:  # 프로그램의 실제 시작 함수입니다.
    motors = MotorTester()  # 모터 테스트용 GPIO 객체를 만듭니다.
    buzzer = Buzzer(BUZZER_PIN)  # 부저 GPIO 객체를 만듭니다.
    sensor = DistanceSensor(  # 초음파 거리 센서 GPIO 객체를 만듭니다.
        echo=ULTRASONIC_ECHO,  # Echo 입력 핀 번호를 설정합니다.
        trigger=ULTRASONIC_TRIG,  # Trig 출력 핀 번호를 설정합니다.
        max_distance=4.0,  # 최대 측정 거리를 4m로 설정합니다.
    )  # DistanceSensor 설정 끝입니다.

    try:  # 프로그램 실행 중 Ctrl+C 같은 상황도 정리하기 위해 try를 사용합니다.
        while True:  # 사용자가 종료를 선택할 때까지 계속 메뉴를 반복합니다.
            print_menu()  # 메뉴를 화면에 출력합니다.
            choice = input("Select: ").strip().lower()  # 사용자 입력을 받고 공백 제거 후 소문자로 바꿉니다.

            if choice == "1":  # 사용자가 1을 입력했는지 확인합니다.
                test_buzzer(buzzer)  # 부저 테스트를 실행합니다.
            elif choice == "2":  # 사용자가 2를 입력했는지 확인합니다.
                test_ultrasonic(sensor)  # 초음파 센서 테스트를 실행합니다.
            elif choice == "3":  # 사용자가 3을 입력했는지 확인합니다.
                test_motors(motors)  # 모터 테스트를 실행합니다.
            elif choice == "4":  # 사용자가 4를 입력했는지 확인합니다.
                test_motors_with_speed(motors)  # 속도제어 모터 테스트를 실행합니다.
            elif choice == "5":  # 사용자가 5를 입력했는지 확인합니다.
                test_buzzer(buzzer)  # 전체 테스트 중 부저 테스트를 먼저 실행합니다.
                test_ultrasonic(sensor)  # 전체 테스트 중 초음파 센서 테스트를 실행합니다.
                test_motors_with_speed(motors)  # 전체 테스트 중 속도제어 모터 테스트를 마지막으로 실행합니다.
            elif choice in {"q", "quit", "exit"}:  # 종료 명령을 입력했는지 확인합니다.
                break  # while 반복문을 빠져나가 프로그램 종료 단계로 갑니다.
            else:  # 위 조건 어디에도 해당하지 않는 입력일 때 실행됩니다.
                print("Unknown selection.")  # 알 수 없는 선택이라고 안내합니다.
    except KeyboardInterrupt:  # 키보드에서 Ctrl+C를 눌렀을 때 실행됩니다.
        print("\nInterrupted.")  # 중단되었다고 출력합니다.
    finally:  # 정상 종료든 오류 종료든 마지막에 반드시 실행됩니다.
        motors.close()  # 모터 GPIO를 정리합니다.
        buzzer.off()  # 혹시 켜져 있을 수 있는 부저를 끕니다.
        buzzer.close()  # 부저 GPIO를 정리합니다.
        sensor.close()  # 초음파 센서 GPIO를 정리합니다.
        print("GPIO cleanup complete.")  # GPIO 정리가 끝났다고 출력합니다.


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래 코드를 실행합니다.
    main()  # main 함수를 실행해서 프로그램을 시작합니다.

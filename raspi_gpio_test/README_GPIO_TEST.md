# Raspberry Pi 5 GPIO Wiring Test

이 폴더는 모터드라이버, 부저, 초음파 센서 배선 테스트에 필요한 파일만 모아둔 폴더입니다.

## 파일

- `gpio_wiring_test.py`: GPIO 배선 테스트 코드
- `requirements_gpio_test.txt`: Python 패키지 목록

## 라즈베리파이에서 실행

```bash
cd raspi_gpio_test
python3 gpio_wiring_test.py
```

패키지가 없다고 나오면 아래 명령을 먼저 실행하세요.

```bash
sudo apt update
sudo apt install -y python3-gpiozero python3-lgpio
```

## 메뉴

```text
1. Test buzzer
2. Test ultrasonic sensor
3. Test motors
4. Test motors with speed control
5. Test all
q. Quit
```

## 주의

초음파 센서가 HC-SR04라면 Echo 핀은 보통 5V입니다.
라즈베리파이 GPIO는 3.3V만 안전하므로 Echo와 GPIO26 사이에 전압분배 저항이나 레벨시프터를 넣어야 합니다.

# Raspberry Pi YOLO TFLite Safety Driving Robot

이 코드는 YOLO로 만든 `.tflite` 모델을 사용해서 카메라 화면의 `head`, `helmet`, `vest`를 탐지하고, 안전 상태에 따라 2륜 또는 4륜 로봇을 전진/회전/정지시키는 예제입니다.

## 실행 준비

라즈베리파이에 이 폴더를 복사하고, YOLO TFLite 모델 파일을 같은 폴더에 `best_float32.tflite` 이름으로 넣습니다. - .pt라면 알아서 맞춰서 넣습니다.

```bash
cd raspi_yolo_tflite_robot
pip install -r requirements_raspi.txt
```

## 2륜 실행

```bash
python run_2wd_yolo_tflite.py
```

## 4륜 실행

```bash
python run_4wd_yolo_tflite.py
```

## 주의

현재 데이터셋이 `head`, `helmet`, `vest`만 포함한다면 진짜 사람 박스가 아니라 안전장비 관련 박스를 기준으로 따라갑니다.

`person` 클래스까지 학습한 YOLO 모델을 쓰면 `config.py`의 `PERSON_CLASS_ID`를 해당 클래스 번호로 바꾸면 됩니다.

txt 파일의 데이터 : 맨앞의 숫자로 판단

0 = head
1 = helmet
2 = vest

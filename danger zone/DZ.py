# =============================================================================
# Safe Eye — 위험 구역 감지 시스템
# danger_zone_detection.py
#
# [역할]
# CCTV(또는 라즈베리파이 카메라)에서 실시간으로 프레임을 받아
# 1) 위험 표지판 위치를 30프레임 마다 갱신
# 2) 매 프레임마다 사람을 감지
# 3) 사람이 표지판 100픽셀 이내에 들어오면 DANGER 경고 표시
#
# [실행 방법]
# VSCode 터미널:
#   python danger_zone_detection.py
#
# Colab:
#   !python danger_zone_detection.py
#
# [필요 파일]
# - config.yaml              : 설정값 파일 (같은 폴더에 있어야 함)
# - danger_sign_yolo11n.pt   : 표지판 탐지 모델 (류상균 팀원한테 받기)
# - best.pt                  : 사람 감지 YOLO 모델 (경로 추가 필요)
# - test_video.mp4           : colab 테스트용 영상 (colab 전용)
#
# [나중에 detector.py 통합 시]
# danger_zone_detection.py 에서 DangerZoneDetector 클래스만 떼어서
# detector.py의 Detector.__init__ 에 추가하면 됩니다.
# =============================================================================

import cv2
import numpy as np
import time
import yaml
from pathlib import Path
from ultralytics import YOLO


# =============================================================================
# 0. 환경 설정
#    ENV = "colab" -> 테스트용 영상 파일(.mp4) 입력, cv2.imshow 없이 파일 저장
#    ENV = "rbp"   -> 라즈베리파이 카메라 실시간 입력, cv2.imshow 출력
# =============================================================================
ENV = "rbp"  # "colab" 또는 "rbp"

# =============================================================================
# 1. 경로 설정
#    [colab 전용]
#    - VIDEO_PATH  : 테스트용 영상 파일 경로 (직접 수정)
#    - OUTPUT_PATH : 결과 영상 저장 경로 (직접 수정)
#
#    [모델 경로]
#    - config.yaml 의 models 섹션에서 관리
#      danger_detector: "danger_sign_yolo11n.pt"  <- 표지판 모델
#      person_detector: "best.pt"                 <- 사람 감지 모델 (경로 추가 필요)
# =============================================================================
CONFIG_PATH  = "config.yaml"        # config 파일 경로 (수정 필요)
# VIDEO_PATH   = "test_video.mp4"     # colab 테스트용 영상 경로 (수정 필요)
OUTPUT_PATH  = "output_danger_zone.mp4"  # 결과 저장 경로 (수정 필요)
SAVE_OUTPUT  = False                # True 로 바꾸면 결과 영상 저장


# =============================================================================
# 2. config.yaml 로드
#    config.yaml 에서 읽어오는 값:
#    - camera.width / height          : 카메라 해상도
#    - models.person_detector         : 사람 감지 모델 경로
#    - models.danger_detector         : 표지판 탐지 모델 경로
#    - thresholds.person_confidence   : 사람 감지 threshold
#    - thresholds.danger_confidence   : 표지판 탐지 threshold
# =============================================================================
def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[Config] 로드 완료: {config_path}")
    return cfg


# =============================================================================
# 3. 위험구역 파라미터
#    config.yaml 에 없는 위험구역 전용 파라미터
#    - DANGER_RADIUS_PX     : 표지판 중심으로부터 위험 반경 (픽셀)
#                             카메라 거리/각도에 따라 현장 튜닝 필요
#    - SIGN_UPDATE_INTERVAL : 표지판 위치 갱신 주기 (프레임)
#                             매 프레임마다 탐지하면 부하가 크므로 30프레임로 설정
# =============================================================================
DANGER_RADIUS_PX     = 100   # 위험 반경 (픽셀) -- 현장 튜닝 필요
SIGN_UPDATE_INTERVAL = 30    # 표지판 갱신 주기 (프레임)


# =============================================================================
# 4. 모델 로드
#    [역할]
#    - person_model : 사람 감지 YOLO 모델 (config의 person_detector)
#    - sign_model   : 표지판 탐지 YOLO 모델 (config의 danger_detector)
#
#    [주의]
#    - danger_detector 경로가 비어있으면 표지판 탐지 비활성화
#    - person_detector 경로가 비어있으면 실행 불가 (필수)
# =============================================================================
def load_models(cfg: dict):
    model_cfg = cfg["models"]

    # 사람 감지 모델 (필수)
    person_path = model_cfg.get("person_detector", "")
    if not person_path:
        raise ValueError("config.yaml 의 models.person_detector 경로를 설정해주세요.")
    print(f"[모델] 사람 감지 모델 로드: {person_path}")
    person_model = YOLO(person_path)

    # 표지판 탐지 모델 (필수)
    danger_path = model_cfg.get("danger_detector", "")
    if not danger_path:
        raise ValueError("config.yaml 의 models.danger_detector 경로를 설정해주세요.")
    print(f"[모델] 표지판 탐지 모델 로드: {danger_path}")
    sign_model = YOLO(danger_path)

    print("[모델] 로드 완료")
    return person_model, sign_model


# =============================================================================
# 5. 유틸리티 함수
# =============================================================================
def get_bbox_center(bbox):
    """bbox 중심 좌표 반환 (x1,y1,x2,y2) -> (cx, cy)"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_bottom_center(bbox):
    """
    사람 bbox 하단 중심 반환
    [이유] 사람의 발 위치 기준으로 표지판과의 거리를 측정
           머리 기준보다 발 기준이 실제 위험 판단에 더 적합
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def calc_distance(pt1, pt2):
    """두 점 사이 유클리드 거리"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def is_in_danger_zone(person_bbox, sign_centers, radius=DANGER_RADIUS_PX):
    """
    사람 발 위치가 표지판 중심에서 radius 픽셀 이내인지 확인
    [입력]
    - person_bbox  : 사람 bbox (x1, y1, x2, y2)
    - sign_centers : 표지판 중심 좌표 리스트 [(cx, cy), ...]
    - radius       : 위험 반경 (픽셀)
    [반환]
    - (True, 가까운 표지판 중심) 또는 (False, None)
    """
    person_pt = get_bbox_bottom_center(person_bbox)
    for sc in sign_centers:
        dist = calc_distance(person_pt, sc)
        if dist <= radius:
            return True, sc
    return False, None


# =============================================================================
# 6. 탐지 함수
# =============================================================================
def detect_signs(frame, model, conf):
    """
    [역할] 프레임에서 위험 표지판 탐지
    [반환] centers (표지판 중심 좌표 리스트), bboxes (표지판 bbox 리스트)
    [호출 시점] 30프레임마다 한 번 호출 -> 결과를 캐시해서 재사용
    """
    results = model(frame, conf=conf, verbose=False)
    centers, bboxes = [], []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centers.append(get_bbox_center((x1, y1, x2, y2)))
            bboxes.append((x1, y1, x2, y2))
    return centers, bboxes


def detect_persons(frame, model, conf):
    """
    [역할] 프레임에서 사람 감지
    [반환] bboxes (사람 bbox 리스트)
    [호출 시점] 매 프레임마다 호출
    [주의] classes=[0] -> YOLO에서 person 클래스만 감지
           best.pt 의 person 클래스 인덱스가 다르면 수정 필요
    """
    results = model(frame, conf=conf, classes=[0], verbose=False)
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxes.append((x1, y1, x2, y2))
    return bboxes


# =============================================================================
# 7. 오버레이 그리기 함수
# =============================================================================
def draw_danger_zones(frame, sign_centers, radius=DANGER_RADIUS_PX):
    """
    [역할] 표지판 중심으로부터 반투명 빨간 원 그리기
    [시각] 반투명(alpha=0.3) 빨간 원 + 불투명 테두리
    """
    overlay = frame.copy()
    for cx, cy in sign_centers:
        cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    for cx, cy in sign_centers:
        cv2.circle(frame, (cx, cy), radius, (0, 0, 220), 2)
    return frame


def draw_sign_bbox(frame, sign_bboxes):
    """[역할] 표지판 bbox + 라벨 그리기 (주황색)"""
    for x1, y1, x2, y2 in sign_bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, "DANGER SIGN",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 165, 255), 2)
    return frame


def draw_persons(frame, person_bboxes, sign_centers, radius=DANGER_RADIUS_PX):
    """
    [역할] 사람 bbox + 상태 라벨 그리기
    [시각]
    - 위험구역 내 : 빨간 bbox + "!! DANGER !!" 텍스트
    - 안전구역    : 초록 bbox + "SAFE" 텍스트
    """
    for bbox in person_bboxes:
        x1, y1, x2, y2 = bbox
        in_danger, _ = is_in_danger_zone(bbox, sign_centers, radius)
        if in_danger:
            color, label = (0, 0, 255), "!! DANGER !!"
        else:
            color, label = (0, 200, 0), "SAFE"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label,
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2)
    return frame


def draw_status(frame, fps, sign_count, frame_count,
                interval=SIGN_UPDATE_INTERVAL):
    """[역할] 화면 상단 상태바 -- FPS, 표지판 수, 다음 갱신까지 남은 시간"""
    next_update = max(0, interval - (frame_count % interval))
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Signs: {sign_count}  |  Next scan: {next_update}f",
                (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


# =============================================================================
# 8. 입력 소스 설정
#    [colab] VIDEO_PATH 에 있는 .mp4 파일을 읽어서 처리
#    [rbp]   라즈베리파이 카메라(인덱스 0) 실시간 입력
#            해상도는 config.yaml 의 camera.width / height 에서 읽어옴
# =============================================================================
def open_capture(cfg: dict):
    cam_cfg = cfg.get("camera", {})
    width  = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)

    if ENV == "colab":
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {VIDEO_PATH}")
        print(f"[입력] 영상 파일: {VIDEO_PATH}")

    elif ENV == "rbp":
        device = cam_cfg.get("device", 0)
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            raise RuntimeError("카메라를 열 수 없습니다. 연결 상태를 확인하세요.")
        print(f"[입력] 라즈베리파이 카메라 연결 완료 (device={device}, {width}x{height})")

    else:
        raise ValueError(f"ENV 값이 올바르지 않습니다: {ENV}  ('colab' 또는 'rbp')")

    print(f"[입력] 소스 FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    return cap


# =============================================================================
# 9. 결과 저장 설정 (선택)
#    SAVE_OUTPUT = True 시 OUTPUT_PATH 에 결과 영상 저장
# =============================================================================
def open_writer(cap):
    if not SAVE_OUTPUT:
        return None
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    print(f"[저장] 결과 영상 저장 경로: {OUTPUT_PATH}")
    return writer


# =============================================================================
# 10. 메인 루프
#     매 프레임 처리 순서:
#     1) 표지판 탐지 (30마다 갱신)
#     2) 사람 감지 (매 프레임)
#     3) 오버레이 그리기
#     4) 화면 출력 / 영상 저장
#
#     [종료]
#     - rbp   : q 키 입력
#     - colab : 영상 파일 끝 또는 강제 중단 (Ctrl+C)
# =============================================================================
def main():
    cfg = load_config(CONFIG_PATH)
    thresh_cfg = cfg.get("thresholds", {})
    person_conf = thresh_cfg.get("person_confidence", 0.5)
    danger_conf = thresh_cfg.get("danger_confidence", 0.45)

    person_model, sign_model = load_models(cfg)
    cap    = open_capture(cfg)
    writer = open_writer(cap)

    # 상태 변수
    sign_centers     = []
    sign_bboxes      = []
    last_sign_update = 0
    fps              = 0.0
    frame_count      = 0

    print("[메인 루프] 시작 -- rbp: q 키로 종료 / colab: 영상 끝까지 실행")

    try:
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[메인 루프] 영상 종료 또는 카메라 연결 끊김")
                break

            frame_count += 1
            t_now = time.time()

            # -- 표지판 탐지 (30프레임 마다) --
            if frame_count % SIGN_UPDATE_INTERVAL == 0:
                sign_centers, sign_bboxes = detect_signs(frame, sign_model, danger_conf)
                last_sign_update = t_now
                print(f"  [표지판 갱신] {len(sign_centers)}개 탐지")

            # -- 사람 감지 (매 프레임) --
            person_bboxes = detect_persons(frame, person_model, person_conf)

            # -- 오버레이 그리기 --
            frame = draw_danger_zones(frame, sign_centers)
            frame = draw_sign_bbox(frame, sign_bboxes)
            frame = draw_persons(frame, person_bboxes, sign_centers)

            elapsed    = t_now - t_start
            fps        = frame_count / elapsed if elapsed > 0 else 0
            frame = draw_status(frame, fps, len(sign_centers), frame_count)

            # -- 결과 저장 --
            if writer:
                writer.write(frame)

            # -- 화면 출력 --
            if ENV == "rbp":
                cv2.imshow("Safe Eye -- Danger Zone", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[메인 루프] 사용자가 종료했습니다.")
                    break

    except KeyboardInterrupt:
        print("[메인 루프] 강제 중단")

    finally:
        cap.release()
        if writer:
            writer.release()
        if ENV == "rbp":
            cv2.destroyAllWindows()
        print(f"[종료] 총 {frame_count}프레임 처리  |  평균 FPS: {fps:.1f}")


if __name__ == "__main__":
    main()

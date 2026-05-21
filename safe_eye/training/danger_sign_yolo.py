from pathlib import Path
import argparse
import shutil

from ultralytics import YOLO


DEFAULT_DATA = Path(r"C:\Users\KCCISTC\Desktop\표지판\데이터셋\data.yaml")
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "danger_sign_yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(description="Train danger sign detector with YOLO.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO data.yaml path")
    parser.add_argument("--model", default="yolo11n.pt", help="base YOLO model")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", default="-1", help="batch size, -1 for auto")
    parser.add_argument("--name", default="danger_sign_yolo11n")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="copy best.pt here")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    batch = int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        patience=20,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    output = Path(args.output)
    shutil.copy2(best_pt, output)
    print("TRAIN_DONE")
    print("run_dir=", run_dir)
    print("best_pt=", best_pt)
    print("copied_to=", output)


if __name__ == "__main__":
    main()

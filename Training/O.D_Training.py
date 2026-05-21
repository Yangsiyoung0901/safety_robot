#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train YOLO11n with the local mixed_22000_plus_checked4000 dataset.

Run:
    python train_yolo11n_mixed_22000_checked_colab.py

Optional examples:
    python train_yolo11n_mixed_22000_checked_colab.py --epochs 50 --imgsz 640 --batch 16
    python train_yolo11n_mixed_22000_checked_colab.py --device cpu --epochs 5
    python train_yolo11n_mixed_22000_checked_colab.py --check-only
    python train_yolo11n_mixed_22000_checked_colab.py --plots-only
    python train_yolo11n_mixed_22000_checked_colab.py --export-onnx
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    SCRIPT_DIR
    / "mixed_22000_plus_checked4000_train_valid_colab"
    / "mixed_22000_plus_checked4000_train_valid_colab"
)
DEFAULT_PROJECT = SCRIPT_DIR / "runs" / "detect"
DEFAULT_RUN_NAME = "PPE_yolo11n_mixed_22000_checked4000"
CLASS_NAMES = ["head", "helmet", "vest"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11n PPE detector.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", default="-1", help="Use -1 for Ultralytics autobatch.")
    parser.add_argument("--device", default=None, help="Examples: 0, 0,1, cpu. Default: auto.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Only make extra EN/KO graphs from results.csv.")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def normalize_batch(value: str) -> int | float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch must be a number, for example -1 or 16") from exc

    if number.is_integer():
        return int(number)
    return number


def write_local_data_yaml(data_root: Path) -> Path:
    data_root = data_root.resolve()
    data_yaml = data_root / "data.yaml"

    train_images = data_root / "train" / "images"
    valid_images = data_root / "valid" / "images"
    train_labels = data_root / "train" / "labels"
    valid_labels = data_root / "valid" / "labels"

    missing = [
        path
        for path in [train_images, valid_images, train_labels, valid_labels]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Dataset folders not found:\n{missing_text}")

    data_yaml.write_text(
        f"path: {data_root.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES!r}\n",
        encoding="utf-8",
    )
    return data_yaml


def count_files(path: Path, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns for _ in path.glob(pattern))


def print_dataset_summary(data_root: Path, data_yaml: Path) -> None:
    image_patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    train_image_count = count_files(data_root / "train" / "images", image_patterns)
    valid_image_count = count_files(data_root / "valid" / "images", image_patterns)
    train_label_count = count_files(data_root / "train" / "labels", ("*.txt",))
    valid_label_count = count_files(data_root / "valid" / "labels", ("*.txt",))

    print(f"DATA_ROOT = {data_root}")
    print(f"DATA_YAML = {data_yaml}")
    print(f"train images = {train_image_count}")
    print(f"train labels = {train_label_count}")
    print(f"valid images = {valid_image_count}")
    print(f"valid labels = {valid_label_count}")


def zip_training_outputs(run_dir: Path) -> Path:
    zip_base = run_dir.parent / f"{run_dir.name}_outputs"
    zip_path = shutil.make_archive(str(zip_base), "zip", run_dir)
    return Path(zip_path)


def read_results_csv(results_csv: Path) -> tuple[list[int], dict[str, list[float]]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    epochs: list[int] = []
    series: dict[str, list[float]] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, raw_row in enumerate(reader):
            row = {key.strip(): value for key, value in raw_row.items() if key is not None}
            epoch_text = row.get("epoch", str(row_index))
            epochs.append(int(float(epoch_text)) + 1)
            for key, value in row.items():
                if key == "epoch" or value in ("", None):
                    continue
                try:
                    series.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return epochs, series


def configure_korean_font() -> None:
    import matplotlib
    from matplotlib import font_manager

    font_candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(font_path)
            ).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_group(
    epochs: list[int],
    series: dict[str, list[float]],
    columns: list[str],
    labels: dict[str, str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    existing_columns = [column for column in columns if column in series]
    if not existing_columns:
        return

    plt.figure(figsize=(11, 6.2), dpi=150)
    for column in existing_columns:
        values = series[column]
        usable_epochs = epochs[: len(values)]
        plt.plot(usable_epochs, values, linewidth=2, label=labels.get(column, column))

    plt.title(title)
    plt.xlabel(labels.get("epoch", "Epoch"))
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def make_extra_plots(run_dir: Path) -> Path:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it first:\n"
            "    pip install matplotlib"
        ) from exc

    configure_korean_font()
    results_csv = run_dir / "results.csv"
    epochs, series = read_results_csv(results_csv)
    output_dir = run_dir / "extra_plots"

    loss_columns = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    metric_columns = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    lr_columns = [column for column in series if column.startswith("lr/")]

    labels_en = {
        "epoch": "Epoch",
        "train/box_loss": "Train box loss",
        "train/cls_loss": "Train class loss",
        "train/dfl_loss": "Train DFL loss",
        "val/box_loss": "Validation box loss",
        "val/cls_loss": "Validation class loss",
        "val/dfl_loss": "Validation DFL loss",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
        "lr/pg0": "LR group 0",
        "lr/pg1": "LR group 1",
        "lr/pg2": "LR group 2",
    }
    labels_ko = {
        "epoch": "에폭",
        "train/box_loss": "학습 박스 손실",
        "train/cls_loss": "학습 클래스 손실",
        "train/dfl_loss": "학습 DFL 손실",
        "val/box_loss": "검증 박스 손실",
        "val/cls_loss": "검증 클래스 손실",
        "val/dfl_loss": "검증 DFL 손실",
        "metrics/precision(B)": "정밀도",
        "metrics/recall(B)": "재현율",
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
        "lr/pg0": "학습률 그룹 0",
        "lr/pg1": "학습률 그룹 1",
        "lr/pg2": "학습률 그룹 2",
    }

    plot_group(
        epochs,
        series,
        loss_columns,
        labels_en,
        "Training and Validation Loss",
        "Loss",
        output_dir / "loss_curves_en.png",
    )
    plot_group(
        epochs,
        series,
        loss_columns,
        labels_ko,
        "학습 및 검증 손실",
        "손실",
        output_dir / "loss_curves_ko.png",
    )
    plot_group(
        epochs,
        series,
        metric_columns,
        labels_en,
        "Validation Metrics",
        "Score",
        output_dir / "metrics_curves_en.png",
    )
    plot_group(
        epochs,
        series,
        metric_columns,
        labels_ko,
        "검증 성능 지표",
        "점수",
        output_dir / "metrics_curves_ko.png",
    )
    plot_group(
        epochs,
        series,
        lr_columns,
        labels_en,
        "Learning Rate",
        "Learning rate",
        output_dir / "learning_rate_en.png",
    )
    plot_group(
        epochs,
        series,
        lr_columns,
        labels_ko,
        "학습률",
        "학습률",
        output_dir / "learning_rate_ko.png",
    )

    print(f"EXTRA_PLOTS = {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    batch = normalize_batch(str(args.batch))
    data_root = args.data_root.resolve()
    data_yaml = write_local_data_yaml(data_root)
    print_dataset_summary(data_root, data_yaml)
    if args.check_only:
        print("check-only complete")
        return

    run_dir = args.project.resolve() / args.name
    if args.plots_only:
        make_extra_plots(run_dir)
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Install it first:\n"
            "    pip install -U ultralytics"
        ) from exc

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": batch,
        "patience": args.patience,
        "project": str(args.project.resolve()),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "workers": args.workers,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    print("Starting training...")
    model.train(**train_kwargs)

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"RUN_DIR = {run_dir}")
    print(f"BEST_PT = {best_pt} exists={best_pt.exists()}")
    print(f"LAST_PT = {last_pt} exists={last_pt.exists()}")

    if (run_dir / "results.csv").exists():
        make_extra_plots(run_dir)

    if not args.no_zip and run_dir.exists():
        zip_path = zip_training_outputs(run_dir)
        print(f"ZIP_OUT = {zip_path}")

    if args.export_onnx:
        if not best_pt.exists():
            raise FileNotFoundError(f"Cannot export ONNX because best.pt was not found: {best_pt}")
        export_model = YOLO(str(best_pt))
        onnx_path = export_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"ONNX = {onnx_path}")


if __name__ == "__main__":
    main()

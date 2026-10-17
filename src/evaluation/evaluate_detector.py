"""Evaluate the selected detector and export aggregate and per-class metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.common.config import load_settings, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def collect_metrics(metrics: Any, names: Any) -> dict[str, Any]:
    per_class = []
    class_indices = [int(value) for value in metrics.box.ap_class_index]

    for result_index, class_id in enumerate(class_indices):
        precision, recall, map50, map50_95 = metrics.box.class_result(result_index)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map50_95": float(map50_95),
            }
        )

    return {
        "aggregate": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline.yaml"
    )
    parser.add_argument("--model", type=Path, help="Override detector checkpoint")
    parser.add_argument("--data", type=Path, help="Override dataset YAML")
    parser.add_argument("--device", help="Override device, for example 0 or cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "detector",
    )
    parser.add_argument("--plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings(args.config)
    model_path = (args.model or settings.detector.model_path).resolve()
    data_yaml = (args.data or settings.detector.data_yaml).resolve()
    device = resolve_device(args.device or settings.detector.device)

    if not model_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {model_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=settings.detector.image_size,
        batch=args.batch,
        device=device,
        conf=settings.detector.confidence,
        iou=settings.detector.nms_iou,
        plots=args.plots,
        project=str(args.output),
        name="ultralytics",
        exist_ok=True,
    )

    report = collect_metrics(metrics, model.names)
    report.update(
        {
            "model_path": str(model_path),
            "data_yaml": str(data_yaml),
            "image_size": settings.detector.image_size,
            "confidence": settings.detector.confidence,
            "nms_iou": settings.detector.nms_iou,
            "device": str(device),
        }
    )

    json_path = args.output / "metrics.json"
    csv_path = args.output / "per_class_metrics.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id", "class_name", "precision", "recall", "map50", "map50_95"
            ],
        )
        writer.writeheader()
        writer.writerows(report["per_class"])

    print(json.dumps(report["aggregate"], indent=2))
    print(f"Per-class metrics: {csv_path}")
    print(f"Full report: {json_path}")


if __name__ == "__main__":
    main()

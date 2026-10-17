"""Typed project configuration loaded from ``configs/pipeline.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


@dataclass(frozen=True)
class DetectorSettings:
    model_path: Path
    data_yaml: Path
    confidence: float
    nms_iou: float
    image_size: int
    device: str


@dataclass(frozen=True)
class SeveritySettings:
    enabled: bool
    model_path: Path
    image_size: int
    classes: tuple[str, ...]


@dataclass(frozen=True)
class OutputSettings:
    directory: Path
    save_images: bool
    save_csv: bool
    save_json: bool


@dataclass(frozen=True)
class PipelineSettings:
    detector: DetectorSettings
    severity: SeveritySettings
    output: OutputSettings
    config_path: Path
    project_root: Path


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required setting: {section}.{key}")
    return mapping[key]


def load_settings(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> PipelineSettings:
    """Load and validate the shared pipeline configuration."""
    root = Path(project_root).resolve()
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH

    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    detector = raw.get("detector", {})
    severity = raw.get("severity", {})
    output = raw.get("output", {})

    confidence = float(_require(detector, "confidence", "detector"))
    nms_iou = float(_require(detector, "nms_iou", "detector"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("detector.confidence must be between 0 and 1")
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("detector.nms_iou must be between 0 and 1")

    classes = tuple(str(item) for item in severity.get("classes", []))
    if not classes:
        raise ValueError("severity.classes must contain at least one class")

    return PipelineSettings(
        detector=DetectorSettings(
            model_path=_resolve_path(
                str(_require(detector, "model_path", "detector")), root
            ),
            data_yaml=_resolve_path(
                str(_require(detector, "data_yaml", "detector")), root
            ),
            confidence=confidence,
            nms_iou=nms_iou,
            image_size=int(_require(detector, "image_size", "detector")),
            device=str(detector.get("device", "auto")),
        ),
        severity=SeveritySettings(
            enabled=bool(severity.get("enabled", True)),
            model_path=_resolve_path(
                str(_require(severity, "model_path", "severity")), root
            ),
            image_size=int(severity.get("image_size", 224)),
            classes=classes,
        ),
        output=OutputSettings(
            directory=_resolve_path(str(output.get("directory", "outputs/inference")), root),
            save_images=bool(output.get("save_images", True)),
            save_csv=bool(output.get("save_csv", True)),
            save_json=bool(output.get("save_json", True)),
        ),
        config_path=path,
        project_root=root,
    )


def resolve_device(requested: str) -> str | int:
    """Translate ``auto`` into the best available Ultralytics device value."""
    if requested.lower() != "auto":
        return int(requested) if requested.isdigit() else requested

    import torch

    return 0 if torch.cuda.is_available() else "cpu"

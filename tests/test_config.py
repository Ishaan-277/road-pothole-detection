from pathlib import Path

import pytest

from src.common.config import load_settings


def write_config(path: Path, confidence: float = 0.2) -> None:
    path.write_text(
        f"""
detector:
  model_path: models/detector.pt
  data_yaml: configs/data.yaml
  confidence: {confidence}
  nms_iou: 0.4
  image_size: 640
  device: cpu
severity:
  enabled: true
  model_path: models/severity.pth
  image_size: 224
  classes: [low, medium, high]
output:
  directory: outputs/test
  save_images: true
  save_csv: true
  save_json: true
""".strip(),
        encoding="utf-8",
    )


def test_load_settings_resolves_project_relative_paths(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.yaml"
    write_config(config)

    settings = load_settings(config, project_root=tmp_path)

    assert settings.detector.model_path == tmp_path / "models" / "detector.pt"
    assert settings.detector.data_yaml == tmp_path / "configs" / "data.yaml"
    assert settings.severity.classes == ("low", "medium", "high")
    assert settings.output.directory == tmp_path / "outputs" / "test"


def test_load_settings_rejects_invalid_confidence(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.yaml"
    write_config(config, confidence=1.2)

    with pytest.raises(ValueError, match="confidence"):
        load_settings(config, project_root=tmp_path)

# AI-Based Road Pothole Detection and Monitoring

This project detects four road-defect classes with YOLO11 and classifies detected potholes by visual severity. The current repository contains a trained detection workflow, evaluation utilities, a manual severity-labelling workflow, and a ResNet18 severity-classification experiment.

The project is currently an ML research prototype. Image detection and severity-model training are implemented; video tracking, cross-frame deduplication, analytics, and the dashboard remain planned.

## Current capabilities

- Detect longitudinal cracks, transverse cracks, alligator cracks, and potholes.
- Train and compare YOLO11n and YOLO11s models.
- Evaluate confidence and non-maximum-suppression thresholds.
- Produce annotated validation samples and CSV detection reports.
- Collect manual low, medium, and high pothole-severity labels.
- Train and evaluate a three-class ResNet18 severity classifier.

See [Project Status](docs/PROJECT_STATUS.md) for the implementation roadmap and [Experiment Log](docs/EXPERIMENT_LOG.md) for the recorded results.

## Dataset

The local dataset uses YOLO-format annotations and is configured in `configs/data.yaml`.

| Split | Images | Bounding boxes |
| --- | ---: | ---: |
| Train | 25,677 | 64,293 |
| Validation | 4,509 | 11,470 |
| Total | 30,186 | 75,763 |

Classes:

| ID | Defect |
| ---: | --- |
| 0 | Longitudinal Crack (D00) |
| 1 | Transverse Crack (D10) |
| 2 | Alligator Crack (D20) |
| 3 | Pothole (D40) |

The dataset and generated model artifacts are intentionally excluded from Git because of their size.

## Repository layout

```text
configs/
  data.yaml                    YOLO dataset configuration
docs/
  PROJECT_STATUS.md            Current implementation and roadmap
  EXPERIMENT_LOG.md            Dataset and model experiment results
src/
  detection/                   YOLO training, evaluation, and inspection
  severity/                    Severity labelling, preparation, and training
  tracking/                    Reserved for video tracking work
  analytics/                   Reserved for reporting and analytics work
data/                          Local images and labels; ignored by Git
outputs/                       Training and evaluation artifacts; ignored by Git
```

## Python setup

Python 3.12 is recommended for this project. A virtual environment has its own package set, so having Python 3.12 installed does not by itself install the packages listed in `requirements.txt`.

```powershell
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Python launcher is unavailable, use the installed interpreter directly:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv venv312
```

## Main workflows

Run the end-to-end image pipeline on one image or a directory:

```powershell
python -m src.pipeline --input path\to\images --output outputs\inspection
```

Useful overrides include `--recursive`, `--device cpu`, `--confidence 0.25`, `--iou 0.40`, and `--no-severity`. The pipeline writes annotated images, `detections.csv`, and a versioned `report.json` containing run configuration and image-level status.

Run video detection, ByteTrack association, and track-level severity aggregation:

```powershell
python -m src.video --input path\to\inspection.mp4 --output outputs\video_inspection
```

The video workflow writes `annotated.mp4`, `tracks.csv`, and a versioned `report.json`. A track represents a unique defect within one continuous clip; it is not yet a cross-route physical-defect identity.

Train the baseline detector:

```powershell
python src/detection/train_yolo11s.py
```

Evaluate detector thresholds and inspect the selected setting:

```powershell
python src/detection/evaluate_thresholds.py
python src/detection/inspect_selected_thresholds.py
```

Build and train the severity classifier:

```powershell
python -m src.severity.collect_labels
python -m src.severity.prepare_dataset
python -m src.severity.prepare_splits
python -m src.severity.train_classifier
```

Most detection scripts currently target NVIDIA GPU device `0`. The severity classifier automatically selects CUDA when available and otherwise uses the CPU.

## Quality checks

Run the fast unit-test suite and dataset-integrity audit:

```powershell
python -m pytest
python -m src.evaluation.dataset_audit
```

Run the selected detector on the validation set and export aggregate and per-class metrics:

```powershell
python -m src.evaluation.evaluate_detector
```

Shared model paths, thresholds, image sizes, output settings, and device selection are defined in `configs/pipeline.yaml`.

## Best recorded results

- Detector: YOLO11s, 640 px, mAP50 `0.6470`, mAP50-95 `0.3422`.
- Severity classifier: test accuracy `61.8%`, balanced accuracy `60.4%` on 68 held-out crops.

These results are research baselines, not production acceptance metrics.

# Project Status

Last updated: 2026-10-16

## Summary

The project currently provides an offline image-based road-defect detector and a prototype pothole-severity classifier. The strongest completed component is the YOLO11s detector. Severity classification is functional but data-limited. Video processing, tracking, unique-defect counting, analytics, and a user-facing dashboard have not yet been implemented.

## Component status

| Component | Status | Notes |
| --- | --- | --- |
| Four-class YOLO dataset | Complete | 30,186 images and 75,763 validated boxes across train and validation splits. |
| YOLO11n baseline | Complete | Trained for 30 epochs. |
| YOLO11s 640 baseline | Complete | Best detector so far. |
| YOLO11s fine-tuning experiments | Complete | Low-learning-rate and 800 px runs did not improve the baseline. |
| Threshold evaluation | Complete | Confidence/NMS grid and visual comparisons generated. |
| Annotation review tooling | Complete | Ground-truth and prediction comparisons generated for sample images. |
| Manual severity labelling | Complete for initial batch | 327 detections reviewed from 100 images. |
| Severity dataset preparation | Complete | 310 usable examples after removing 17 duplicates. |
| ResNet18 severity baseline | Complete | Initial three-class model trained and evaluated. |
| End-to-end image inference | Not started | Detector and severity model are not yet connected in one command. |
| Video ingestion and processing | Not started | No video pipeline currently exists. |
| Multi-frame tracking | Not started | `src/tracking` is reserved for this work. |
| Unique-defect deduplication | Not started | Current duplicate handling is manual within labelled crops. |
| Road-maintenance analytics | Not started | `src/analytics` is reserved for this work. |
| Streamlit dashboard | Not started | Streamlit is installed but no application exists. |
| Exportable inspection reports | Not started | Current output is limited to images, CSV files, and training plots. |
| Shared pipeline configuration | Complete | Model paths, thresholds, image sizes, output settings, and device selection are centralized. |
| Dataset integrity audit | Complete | Validates pairs, rows, ranges, classes, and duplicate annotations; optional cross-split hashing is available. |
| Per-class detector evaluation | Complete | Exports aggregate and per-class CSV/JSON metrics. |
| Automated tests | Foundation complete | Six fast tests cover configuration and dataset-audit logic; inference tests remain planned. |

## Current model choices

The downstream detection and severity-labelling scripts use:

- Checkpoint: `outputs/training/yolo11s_640_baseline/weights/best.pt`
- Input size: 640 px
- Operational confidence threshold: 0.20
- Operational NMS IoU threshold: 0.40

The threshold sweep achieved its highest mAP50-95 at confidence `0.05` and IoU `0.70`. The stricter operational setting was subsequently used to reduce low-confidence and overlapping detections. This tradeoff should be validated against a task-specific acceptance set before deployment.

At the operational thresholds, D40 pothole performance on the current validation set is precision `0.7624`, recall `0.6579`, mAP50 `0.6322`, and mAP50-95 `0.3347`.

## Known limitations

- The dataset source, licence information, transformation history, and split-generation procedure are not yet documented.
- The detector configuration contains a machine-specific local data path.
- Most detection scripts explicitly select GPU device `0`.
- The severity dataset has only 310 usable crops and is class-imbalanced.
- Severity labels reflect visual judgement and have no recorded multi-reviewer agreement process.
- The severity model is saved as a state dictionary, but no production inference loader exists.
- Model artifacts and datasets are local-only and cannot be reconstructed from the repository.
- Dependency versions are not yet locked to exact releases.

## Recommended next milestones

1. Add an end-to-end image inference command that emits detections and severity predictions.
2. Extend automated tests to crop creation, model adapters, and report generation.
3. Document dataset provenance and provide a reproducible dataset preparation/splitting script.
4. Expand and independently review the severity dataset.
5. Define detector and severity acceptance metrics on a fixed holdout set.
6. Add video inference and a tracker such as ByteTrack.
7. Implement spatial and temporal deduplication for unique pothole counting.
8. Build analytics, reporting, and the Streamlit interface after the inference pipeline stabilizes.

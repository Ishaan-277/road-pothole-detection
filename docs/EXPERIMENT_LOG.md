# Experiment Log

Last updated: 2026-10-16

This document records the completed experiments represented by the local artifacts under `outputs/`. Generated weights, plots, crops, and reports are excluded from Git.

## Detection dataset

| Split | Images | Boxes | D00 | D10 | D20 | D40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 25,677 | 64,294 | 27,114 | 16,358 | 9,941 | 10,881 |
| Validation | 4,509 | 11,471 | 4,807 | 2,992 | 1,769 | 1,903 |
| Total | 30,186 | 75,765 | 31,921 | 19,350 | 11,710 | 12,784 |

Dataset integrity checks found matching image/label counts, no empty label files, and no malformed YOLO rows in either split.

## Detector training

| Run | Completed epochs | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO11n baseline, 640 px | 30 | 30 | 0.6682 | 0.5455 | 0.6040 | 0.3089 | Baseline completed |
| YOLO11s baseline, 640 px | 30 | 30 | 0.6979 | 0.5808 | 0.6470 | 0.3422 | Best run |
| YOLO11s low-LR fine-tune, 640 px | 13 | 1 | 0.6511 | 0.5684 | 0.6146 | 0.3157 | No improvement; early-stopped |
| YOLO11s fine-tune, 800 px | 11 | 1 | 0.6556 | 0.5619 | 0.6109 | 0.3078 | No improvement; early-stopped |

The YOLO11s 640 px baseline remains the selected detector.

## Threshold evaluation

Sixteen validation combinations were evaluated across confidence thresholds `0.05`, `0.10`, `0.15`, and `0.20`, and NMS IoU thresholds `0.40`, `0.50`, `0.60`, and `0.70`.

Best measured mAP50-95 configuration:

| Confidence | NMS IoU | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.70 | 0.6984 | 0.5815 | 0.6267 | 0.3321 |

Selected operational inspection configuration:

| Confidence | NMS IoU | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.40 | 0.6941 | 0.6098 | 0.5615 | 0.2988 |

On the 20-image NMS inspection sample, IoU `0.70` produced 43 boxes while IoU `0.50` and `0.40` each produced 38. On a separate 30-image inspection sample, the selected `0.20`/`0.40` configuration produced 69 detections, with two images receiving no detections.

## Severity dataset

The completed review batch contains 327 detections from 100 source images.

| Label | Count |
| --- | ---: |
| Low | 108 |
| Medium | 148 |
| High | 54 |
| Duplicate/excluded | 17 |
| Usable total | 310 |

The classifier split is grouped by source image so crops from one source image cannot appear in multiple train, validation, or test partitions.

## Severity classifier

Architecture: ImageNet-pretrained ResNet18 with a three-class output layer.

- Best epoch: 4 of 15
- Best validation loss: 0.8084
- Held-out test examples: 68
- Test accuracy: 0.6176
- Test balanced accuracy: 0.6038

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.5238 | 0.5238 | 0.5238 | 21 |
| Medium | 0.6875 | 0.6875 | 0.6875 | 32 |
| High | 0.6000 | 0.6000 | 0.6000 | 15 |

The result establishes a baseline but is not sufficient for production use because the labelled dataset and test set are small.

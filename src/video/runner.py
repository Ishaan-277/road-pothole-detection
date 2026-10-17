"""Video inference with ByteTrack and track-level severity aggregation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.common.config import PipelineSettings, resolve_device
from src.pipeline.io import draw_severity_label, padded_crop
from src.pipeline.models import SeverityPredictor
from src.video.tracks import TrackSummary


POTHOLE_CLASS_ID = 3
TRACK_FIELDS = [
    "track_id",
    "class_id",
    "class_name",
    "first_frame",
    "last_frame",
    "first_time_seconds",
    "last_time_seconds",
    "observations",
    "max_detection_confidence",
    "severity",
    "severity_confidence",
    "severity_observations",
]


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


def run_video(
    input_path: Path,
    settings: PipelineSettings,
    *,
    output_dir: Path,
    severity_enabled: bool | None = None,
) -> dict:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Video not found: {input_path}")
    if not settings.detector.model_path.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {settings.detector.model_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*settings.video.codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_video}")

    device = resolve_device(settings.detector.device)
    detector = YOLO(str(settings.detector.model_path))
    use_severity = (
        settings.severity.enabled if severity_enabled is None else severity_enabled
    )
    severity = (
        SeverityPredictor(
            settings.severity.model_path,
            settings.severity.classes,
            settings.severity.image_size,
            device,
        )
        if use_severity
        else None
    )
    tracks: dict[int, TrackSummary] = {}
    untracked_detections = 0
    processed_frames = 0
    started_at = datetime.now(timezone.utc)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = processed_frames
            processed_frames += 1

            result = detector.track(
                source=frame,
                persist=True,
                tracker=settings.video.tracker,
                imgsz=settings.detector.image_size,
                conf=settings.detector.confidence,
                iou=settings.detector.nms_iou,
                device=device,
                save=False,
                verbose=False,
            )[0]
            annotated = result.plot()
            boxes = result.boxes

            if boxes is not None and len(boxes):
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                for box_index, box in enumerate(boxes):
                    if box_index >= len(ids):
                        untracked_detections += 1
                        continue
                    track_id = int(ids[box_index])
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    coordinates = tuple(
                        int(round(value)) for value in box.xyxy[0].tolist()
                    )
                    summary = tracks.get(track_id)
                    if summary is None:
                        summary = TrackSummary(
                            track_id=track_id,
                            class_id=class_id,
                            class_name=_class_name(detector.names, class_id),
                            first_frame=frame_index,
                            last_frame=frame_index,
                        )
                        tracks[track_id] = summary
                    summary.update_detection(frame_index, confidence)

                    if (
                        class_id == POTHOLE_CLASS_ID
                        and severity is not None
                        and summary.should_sample_severity(
                            frame_index, settings.video.severity_interval_frames
                        )
                    ):
                        crop = padded_crop(frame, coordinates)
                        if crop.size:
                            _, _, probabilities = severity.predict(crop)
                            summary.update_severity(frame_index, probabilities)

                    severity_label, severity_confidence = summary.severity()
                    if severity_label:
                        draw_severity_label(
                            annotated,
                            coordinates,
                            severity_label,
                            severity_confidence,
                        )

            writer.write(annotated)
            if processed_frames % 100 == 0:
                print(
                    f"Frames: {processed_frames}/{frame_count or '?'} | "
                    f"tracks: {len(tracks)}"
                )
    finally:
        capture.release()
        writer.release()

    track_rows = [
        tracks[track_id].as_dict(fps) for track_id in sorted(tracks)
    ]
    tracks_csv = output_dir / "tracks.csv"
    with tracks_csv.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=TRACK_FIELDS)
        writer_csv.writeheader()
        writer_csv.writerows(track_rows)

    report = {
        "schema_version": 1,
        "run": {
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "input_video": str(input_path),
            "output_video": str(output_video),
            "fps": fps,
            "frames": processed_frames,
            "duration_seconds": processed_frames / fps if fps else None,
            "tracks": len(track_rows),
            "unique_pothole_tracks": sum(
                row["class_id"] == POTHOLE_CLASS_ID for row in track_rows
            ),
            "untracked_detections": untracked_detections,
        },
        "configuration": {
            "detector_model": str(settings.detector.model_path),
            "severity_model": str(settings.severity.model_path) if use_severity else None,
            "severity_enabled": use_severity,
            "device": str(device),
            "tracker": settings.video.tracker,
            "severity_interval_frames": settings.video.severity_interval_frames,
            "confidence": settings.detector.confidence,
            "nms_iou": settings.detector.nms_iou,
        },
        "tracks": track_rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

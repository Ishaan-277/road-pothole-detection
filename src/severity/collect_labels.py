
import csv
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "outputs" / "training" / "yolo11s_640_baseline"
    / "weights" / "best.pt"
)
VAL_IMAGES = ROOT / "data" / "images" / "val"

OUTPUT_DIR = ROOT / "outputs" / "pothole_review_batch3"
CROPS_DIR = OUTPUT_DIR / "crops"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CSV_PATH = OUTPUT_DIR / "pothole_labels.csv"

TARGET_IMAGES = 100
CONFIDENCE = 0.20
NMS_IOU = 0.40
IMAGE_SIZE = 640
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FIELDNAMES = [
    "detection_id",
    "image_name",
    "image_path",
    "crop_path",
    "confidence",
    "x1", "y1", "x2", "y2",
    "manual_label",
    "severity_score",
    "duplicate_of",
    "notes",
]

LABELS = {
    ord("1"): ("low", 1),
    ord("2"): ("medium", 2),
    ord("3"): ("high", 3),
    ord("4"): ("duplicate", ""),
    ord("5"): ("uncertain", ""),
    ord("6"): ("not_pothole", ""),
}


def save_csv(rows):
    """Save progress after every label, including partial sessions."""
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VAL_IMAGES.exists():
        raise FileNotFoundError(f"Validation folder not found: {VAL_IMAGES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally replacing an earlier review session.
    if CSV_PATH.exists():
        raise FileExistsError(
            f"{CSV_PATH} already exists.\n"
            "Rename or back up that file before starting a fresh batch."
        )

    image_paths = [
        p for p in VAL_IMAGES.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.seed(RANDOM_SEED)
    random.shuffle(image_paths)

    if not image_paths:
        raise FileNotFoundError("No validation images found.")

    print(f"Searching for {TARGET_IMAGES} images with pothole detections...")
    model = YOLO(str(MODEL_PATH))

    # Store every detection from each selected image.
    review_items = []
    selected_images = 0
    images_checked = 0
    detection_number = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        images_checked += 1

        result = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            classes=[3],
            verbose=False,
        )[0]

        boxes = result.boxes

        # Only count images that have at least one detected pothole.
        if len(boxes) == 0:
            if images_checked % 50 == 0:
                print(
                    f"Checked {images_checked} images; "
                    f"found {selected_images}/{TARGET_IMAGES} qualifying images."
                )
            continue

        selected_images += 1
        annotated = result.plot()
        image_height, image_width = image.shape[:2]
        image_relative_path = str(image_path.relative_to(ROOT))

        annotated_path = ANNOTATED_DIR / image_path.name
        cv2.imwrite(str(annotated_path), annotated)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            confidence = float(box.conf[0].cpu())

            pad_x = max(10, int((x2 - x1) * 0.20))
            pad_y = max(10, int((y2 - y1) * 0.20))

            left = max(0, x1 - pad_x)
            top = max(0, y1 - pad_y)
            right = min(image_width, x2 + pad_x)
            bottom = min(image_height, y2 + pad_y)

            crop = image[top:bottom, left:right]
            if crop.size == 0:
                continue

            detection_number += 1
            detection_id = f"P{detection_number:05d}"
            crop_path = CROPS_DIR / f"{detection_id}.jpg"

            if not cv2.imwrite(str(crop_path), crop):
                print(f"Could not save {crop_path}")
                continue

            review_items.append({
                "detection_id": detection_id,
                "image_name": image_path.name,
                "image_path": image_relative_path,
                "crop_path": str(crop_path.relative_to(ROOT)),
                "confidence": round(confidence, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "manual_label": "",
                "severity_score": "",
                "duplicate_of": "",
                "notes": "",
                # Internal review information; not written to CSV.
                "_image": image,
                "_box_index": len(review_items),
            })

        print(
            f"Qualifying images: {selected_images}/{TARGET_IMAGES} | "
            f"Images checked: {images_checked} | "
            f"Detections collected: {detection_number}"
        )

        if selected_images >= TARGET_IMAGES:
            break

    if selected_images < TARGET_IMAGES:
        print(
            f"\nOnly {selected_images} images with detections were found "
            f"after checking {images_checked} images."
        )

    if not review_items:
        print("No pothole detections found. Nothing to label.")
        return

    # Create clean CSV rows and save a blank review file before labelling.
    rows = [
        {key: item[key] for key in FIELDNAMES}
        for item in review_items
    ]
    save_csv(rows)

    print(f"\nFound {selected_images} images with pothole detections.")
    print(f"Total individual detections to review: {len(review_items)}")
    print("\nStarting keyboard labelling...")
    print(
        "1=Low | 2=Medium | 3=High | 4=Duplicate | "
        "5=Uncertain | 6=Not pothole"
    )
    print("N=Skip | Q=Quit and save")

    window_name = "Pothole Severity Labelling"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for index, item in enumerate(review_items):
            if rows[index]["manual_label"]:
                continue

            image = item["_image"].copy()
            x1, y1, x2, y2 = (
                item["x1"], item["y1"], item["x2"], item["y2"]
            )

            # Highlight the current detection so you know which
            # pothole you are assigning a label to.
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)

            instructions = [
                f"Detection {index + 1}/{len(review_items)}: {item['detection_id']}",
                f"Image: {item['image_name']}",
                f"Model confidence: {item['confidence']:.2f}",
                "1 Low | 2 Medium | 3 High",
                "4 Duplicate | 5 Uncertain | 6 Not pothole",
                "N Skip | Q Quit and save",
            ]

            y = 30
            for line in instructions:
                cv2.putText(
                    image, line, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    image, line, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA
                )
                y += 28

            cv2.imshow(window_name, image)
            key = cv2.waitKey(0) & 0xFF
            key = ord(chr(key).lower()) if key < 128 else key

            if key == ord("q"):
                save_csv(rows)
                print("Stopped. Your progress has been saved.")
                break

            if key == ord("n"):
                rows[index]["manual_label"] = ""
                rows[index]["notes"] = "Skipped during first review"
                save_csv(rows)
                continue

            if key in LABELS:
                label, score = LABELS[key]
                rows[index]["manual_label"] = label
                rows[index]["severity_score"] = score

                if label == "duplicate":
                    print(
                        f"{item['detection_id']} marked duplicate. "
                        "You can fill duplicate_of later if needed."
                    )

                save_csv(rows)
                print(f"Saved {item['detection_id']}: {label}")

        else:
            print("All detections have been reviewed.")

    finally:
        cv2.destroyAllWindows()
        save_csv(rows)

    print(f"\nCSV saved to: {CSV_PATH}")
    print(f"Review crops saved in: {CROPS_DIR}")
    print(f"Annotated images saved in: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()


from pathlib import Path
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "configs" / "data.yaml"

MODEL_PATH = (
    ROOT
    / "outputs"
    / "training"
    / "yolo11s_640_baseline"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = ROOT / "outputs" / "training"

# Training configuration
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 12
DEVICE = 0

# Lower learning rate for fine-tuning
LEARNING_RATE = 0.0001
INITIAL_LR_FACTOR = 0.01

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {MODEL_PATH}"
        )

    print("=" * 55)
    print("YOLO11s 640px - Low Learning Rate Fine-tuning")
    print("=" * 55)
    print(f"Starting weights: {MODEL_PATH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    model = YOLO(str(MODEL_PATH))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Fine-tuning settings
        lr0=LEARNING_RATE,
        lrf=INITIAL_LR_FACTOR,
        warmup_epochs=3,
        weight_decay=0.0005,

        # Reproducibility and checkpoint safety
        seed=42,
        deterministic=True,
        patience=12,
        save=True,
        plots=True,

        # New output directory; preserve the baseline
        project=str(OUTPUT_DIR),
        name="yolo11s_640_lowlr",
        exist_ok=False,
    )

    print("\nTraining finished.")
    print(
        "Results directory:",
        OUTPUT_DIR / "yolo11s_640_lowlr"
    )


if __name__ == "__main__":
    main()

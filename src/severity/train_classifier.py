
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "severity_classifier" / "severity_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "severity_classifier" / "resnet18"
MODEL_PATH = OUTPUT_DIR / "best_severity_model.pth"

CLASSES = ["low", "medium", "high"]
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 16
EPOCHS = 15
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PotholeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = ROOT / row["crop_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = LABEL_TO_ID[row["manual_label"]]
        return image, label


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["manual_label"] = df["manual_label"].str.lower().str.strip()
    df = df[df["manual_label"].isin(CLASSES)].copy()

    missing = [
        path for path in df["crop_path"]
        if not (ROOT / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} crop images are missing. "
            f"First missing path: {missing[0]}"
        )

    # Keep every detection from a source image in the same split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    train_idx, test_idx = next(
        splitter.split(df, df["manual_label"], groups=df["image_name"])
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    assert set(train_df["image_name"]).isdisjoint(set(test_df["image_name"]))

    # Create a validation split from training images, also grouped by image.
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED + 1,
    )
    train_sub_idx, val_idx = next(
        val_splitter.split(
            train_df,
            train_df["manual_label"],
            groups=train_df["image_name"],
        )
    )

    val_df = train_df.iloc[val_idx].copy()
    train_df = train_df.iloc[train_sub_idx].copy()

    assert set(train_df["image_name"]).isdisjoint(set(val_df["image_name"]))
    assert set(val_df["image_name"]).isdisjoint(set(test_df["image_name"]))
    assert set(train_df["image_name"]).isdisjoint(set(test_df["image_name"]))

    print(f"Device: {DEVICE}")
    print(f"Total examples: {len(df)}")
    print(
        f"Train: {len(train_df)} examples from "
        f"{train_df['image_name'].nunique()} images"
    )
    print(
        f"Validation: {len(val_df)} examples from "
        f"{val_df['image_name'].nunique()} images"
    )
    print(
        f"Test: {len(test_df)} examples from "
        f"{test_df['image_name'].nunique()} images"
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(
            brightness=0.12, contrast=0.12, saturation=0.08
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = PotholeDataset(train_df, train_transform)
    val_dataset = PotholeDataset(val_df, eval_transform)
    test_dataset = PotholeDataset(test_df, eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ImageNet-pretrained ResNet18, adapted to our three severity classes.
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained ResNet18 weights. "
            "Check your internet connection and retry."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    # Class-weighted loss helps account for fewer high-severity examples.
    counts = train_df["manual_label"].value_counts()
    class_weights = torch.tensor(
        [
            len(train_df) / (len(CLASSES) * counts.get(name, 1))
            for name in CLASSES
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Evaluate the selected checkpoint on the held-out test images.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Test balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.4f}"
    )
    print("\nTest classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "test_accuracy": accuracy_score(y_true, y_pred),
                "test_balanced_accuracy": balanced_accuracy_score(
                    y_true, y_pred
                ),
                "classification_report": report,
                "classes": CLASSES,
            },
            f,
            indent=2,
        )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASSES
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Pothole Severity Classifier — Test Set")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {OUTPUT_DIR / 'test_metrics.json'}")
    print(f"Confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

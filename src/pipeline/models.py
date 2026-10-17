"""Model adapters used by the inference pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class SeverityPredictor:
    """Load the trained ResNet18 checkpoint and classify BGR crops."""

    def __init__(
        self,
        model_path: Path,
        classes: tuple[str, ...],
        image_size: int,
        device: str | int,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Severity checkpoint not found: {model_path}")

        self.model_path = model_path
        self.classes = classes
        self.device = torch.device(
            f"cuda:{device}" if isinstance(device, int) else device
        )

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(classes))
        try:
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, crop_bgr) -> tuple[str, float, dict[str, float]]:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Cannot classify an empty crop")

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()

        class_id = int(probabilities.argmax().item())
        scores = {
            name: float(probabilities[index].item())
            for index, name in enumerate(self.classes)
        }
        return self.classes[class_id], scores[self.classes[class_id]], scores


from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from segmentation.unet import UNet

IMAGE_DIR = PROJECT / "data/processed/images/2d"
MASK_DIR = PROJECT / "data/processed/masks/2d"
SPLIT_DIR = PROJECT / "data/processed/splits"
MODEL_DIR = PROJECT / "outputs/models"


class BrainMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        split_path = Path(split_file)
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        names = [
            line.strip()
            for line in split_path.read_text().splitlines()
            if line.strip()
        ]

        self.image_files = [
            self.image_dir / f"{name}.npy" for name in names
        ]

        for image_path in self.image_files:
            mask_path = self.mask_dir / image_path.name
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        mask_path = self.mask_dir / image_path.name

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        mask = (mask > 0).astype(np.float32)

        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )


def calculate_metrics(logits, masks):
    predictions = torch.sigmoid(logits) >= 0.5
    targets = masks >= 0.5

    intersection = (predictions & targets).sum().item()
    predicted_pixels = predictions.sum().item()
    target_pixels = targets.sum().item()

    dice = (2 * intersection) / (
        predicted_pixels + target_pixels + 1e-8
    )
    union = (predictions | targets).sum().item()
    iou = intersection / (union + 1e-8)

    return dice, iou


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_dataset = BrainMRIDataset(
        IMAGE_DIR, MASK_DIR, SPLIT_DIR / "train.txt"
    )
    val_dataset = BrainMRIDataset(
        IMAGE_DIR, MASK_DIR, SPLIT_DIR / "validation.txt"
    )

    if set(train_dataset.image_files) & set(val_dataset.image_files):
        raise ValueError("Training and validation slices overlap.")

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, num_workers=0
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = -1.0
    best_model_path = MODEL_DIR / "unet_baseline_best.pth"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Training slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))
    print("Epochs: 5")

    for epoch in range(5):
        model.train()
        total_train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        total_intersection = 0
        total_predicted = 0
        total_target = 0
        total_union = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                total_val_loss += loss.item()

                predictions = torch.sigmoid(logits) >= 0.5
                targets = masks >= 0.5

                total_intersection += (predictions & targets).sum().item()
                total_predicted += predictions.sum().item()
                total_target += targets.sum().item()
                total_union += (predictions | targets).sum().item()

        average_val_loss = total_val_loss / len(val_loader)
        val_dice = (2 * total_intersection) / (
            total_predicted + total_target + 1e-8
        )
        val_iou = total_intersection / (total_union + 1e-8)

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {average_val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}",
            flush=True,
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.cpu().state_dict(), best_model_path)
            model.to(device)
            print("Saved best model:", best_model_path)

    print("Training and validation complete.")
    print("Best validation Dice:", round(best_dice, 4))
    print(
        "Caution: validation slices come from the same patient; "
        "these are not independent patient-level test results."
    )


if __name__ == "__main__":
    main()

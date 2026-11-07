
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from segmentation.unet import UNet


IMAGE_DIR = PROJECT / "data/processed/images/2d"
MASK_DIR = PROJECT / "data/processed/masks/2d"

# Use the best model saved during training and validation.
MODEL_PATH = PROJECT / "outputs/models/unet_baseline_best.pth"

OUTPUT_DIR = PROJECT / "outputs/predictions"


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    image_files = sorted(IMAGE_DIR.glob("*.npy"))

    if not image_files:
        raise FileNotFoundError(
            f"No MRI slices found in {IMAGE_DIR}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = UNet()
    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    evaluated_count = 0
    comparison_data = None

    print("Device:", device)
    print("Model loaded:", MODEL_PATH.name)
    print("Slices to process:", len(image_files))

    for index, image_path in enumerate(image_files, start=1):
        image = np.load(image_path).astype(np.float32)

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)

            predicted_mask = (
                probabilities >= 0.5
            ).squeeze().cpu().numpy().astype(np.uint8)

        prediction_path = (
            OUTPUT_DIR / f"{image_path.stem}_prediction.npy"
        )
        np.save(prediction_path, predicted_mask)
        evaluated_count += 1

        if image_path.stem.endswith("slice_070"):
            true_path = MASK_DIR / image_path.name

            if not true_path.is_file():
                raise FileNotFoundError(
                    f"Ground-truth mask not found: {true_path}"
                )

            true_mask = np.load(true_path)
            true_mask = (true_mask > 0).astype(np.uint8)

            comparison_data = (
                image,
                true_mask,
                predicted_mask,
                image_path.stem,
            )

        if index % 10 == 0 or index == len(image_files):
            print(
                f"Processed {index}/{len(image_files)} slices",
                flush=True,
            )

    if comparison_data is not None:
        image, true_mask, predicted_mask, slice_name = comparison_data

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input MRI")

        axes[1].imshow(true_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Annotated Mask")

        axes[2].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Predicted Mask")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        figure_path = OUTPUT_DIR / f"{slice_name}_comparison.png"
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("Comparison image saved:", figure_path)

    print("Predictions generated:", evaluated_count)
    print("Prediction folder:", OUTPUT_DIR)
    print("Prediction generation complete.")


if __name__ == "__main__":
    main()

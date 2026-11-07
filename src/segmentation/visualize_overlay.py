
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
SLICE_NAME = "UPENN-GBM-00002_11_slice_070"

IMAGE_PATH = (
    PROJECT / "data/processed/images/2d" / f"{SLICE_NAME}.npy"
)
MASK_PATH = (
    PROJECT / "data/processed/masks/2d" / f"{SLICE_NAME}.npy"
)
PREDICTION_PATH = (
    PROJECT / "outputs/predictions" / f"{SLICE_NAME}_prediction.npy"
)
OUTPUT_PATH = (
    PROJECT / "outputs/predictions" / f"{SLICE_NAME}_overlay.png"
)


def main():
    for path in (IMAGE_PATH, MASK_PATH, PREDICTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    image = np.load(IMAGE_PATH)
    true_mask = np.load(MASK_PATH) > 0
    pred_mask = np.load(PREDICTION_PATH) > 0

    if image.shape != true_mask.shape or image.shape != pred_mask.shape:
        raise ValueError("Image, annotated mask, and prediction shapes differ.")

    # Normalize the MRI image for display.
    low, high = np.percentile(image, [1, 99])
    display_image = np.clip(
        (image - low) / (high - low + 1e-8), 0, 1
    )

    # Build a colour overlay:
    # Green = annotated mask only
    # Red = predicted mask only
    # Yellow = overlap between annotated and predicted masks
    overlay = np.stack(
        [display_image, display_image, display_image], axis=-1
    )

    true_only = true_mask & ~pred_mask
    pred_only = pred_mask & ~true_mask
    overlap = true_mask & pred_mask

    overlay[true_only] = [0, 1, 0]
    overlay[pred_only] = [1, 0, 0]
    overlay[overlap] = [1, 1, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(display_image, cmap="gray")
    axes[0].set_title("Input MRI")
    axes[1].imshow(overlay)
    axes[1].set_title("Mask Comparison Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle("U-Net Tumour Segmentation — Slice 070")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Overlay saved:", OUTPUT_PATH)
    print("Green = annotated mask only")
    print("Red = predicted mask only")
    print("Yellow = overlap between both masks")


if __name__ == "__main__":
    main()

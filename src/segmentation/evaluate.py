
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT / "data/processed/images/2d"
MASK_DIR = PROJECT / "data/processed/masks/2d"
PREDICTION_DIR = PROJECT / "outputs/predictions"
SPLIT_FILE = PROJECT / "data/processed/splits/validation.txt"
REPORT_FILE = PROJECT / "outputs/validation_evaluation_report.txt"


def calculate_metrics(true_mask, pred_mask):
    true_mask = true_mask > 0
    pred_mask = pred_mask > 0

    if true_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Shape mismatch: {true_mask.shape} vs {pred_mask.shape}"
        )

    intersection = int(np.logical_and(true_mask, pred_mask).sum())
    union = int(np.logical_or(true_mask, pred_mask).sum())
    true_pixels = int(true_mask.sum())
    pred_pixels = int(pred_mask.sum())

    dice_denominator = true_pixels + pred_pixels
    dice = (
        2 * intersection / dice_denominator
        if dice_denominator > 0
        else 1.0
    )
    iou = intersection / union if union > 0 else 1.0

    return dice, iou, intersection, union, true_pixels, pred_pixels


def main():
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError(f"Validation split not found: {SPLIT_FILE}")

    slice_names = [
        line.strip()
        for line in SPLIT_FILE.read_text().splitlines()
        if line.strip()
    ]

    if not slice_names:
        raise RuntimeError("The validation split file is empty.")

    dice_scores = []
    iou_scores = []
    total_intersection = 0
    total_union = 0
    total_true_pixels = 0
    total_pred_pixels = 0
    lines = []

    print("Evaluating validation slices only...")
    print("Model predictions:", PREDICTION_DIR)
    print("Validation split:", SPLIT_FILE)
    print()

    for slice_name in slice_names:
        image_path = IMAGE_DIR / f"{slice_name}.npy"
        true_path = MASK_DIR / f"{slice_name}.npy"
        pred_path = PREDICTION_DIR / f"{slice_name}_prediction.npy"

        for path in (image_path, true_path, pred_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required file missing: {path}")

        true_mask = np.load(true_path)
        pred_mask = np.load(pred_path)

        dice, iou, intersection, union, true_pixels, pred_pixels = (
            calculate_metrics(true_mask, pred_mask)
        )

        dice_scores.append(dice)
        iou_scores.append(iou)
        total_intersection += intersection
        total_union += union
        total_true_pixels += true_pixels
        total_pred_pixels += pred_pixels

        line = f"{slice_name}: Dice={dice:.4f}, IoU={iou:.4f}"
        lines.append(line)
        print(line)

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))

    dice_denominator = total_true_pixels + total_pred_pixels
    overall_dice = (
        2 * total_intersection / dice_denominator
        if dice_denominator > 0
        else 1.0
    )
    overall_iou = (
        total_intersection / total_union
        if total_union > 0
        else 1.0
    )

    summary = [
        "",
        "========== VALIDATION EVALUATION SUMMARY ==========",
        f"Validation slices evaluated: {len(slice_names)}",
        f"Mean Dice: {mean_dice:.4f} ({mean_dice * 100:.2f}%)",
        f"Mean IoU: {mean_iou:.4f} ({mean_iou * 100:.2f}%)",
        f"Overall Dice: {overall_dice:.4f} ({overall_dice * 100:.2f}%)",
        f"Overall IoU: {overall_iou:.4f} ({overall_iou * 100:.2f}%)",
        "",
        "Caution: validation slices come from the same patient as training.",
        "These results are not independent patient-level test results.",
    ]

    for line in summary:
        print(line)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines + summary) + "\n")
    print("\nReport saved:", REPORT_FILE)


if __name__ == "__main__":
    main()

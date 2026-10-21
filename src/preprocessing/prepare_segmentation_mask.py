
from pathlib import Path

import nibabel as nib
import numpy as np

DATA_DIR = (
    Path.home()
    / "Documents"
    / "BrainTumorDataset"
    / "NIfTI"
)

MASK_PATH = (
    DATA_DIR
    / "Segmentation"
    / "UPENN-GBM-00002_11_segm.nii"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "processed"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_img = nib.load(str(MASK_PATH))
    mask = np.asarray(mask_img.dataobj)

    if not np.isfinite(mask).all():
        raise ValueError("Mask contains non-finite values.")

    labels, counts = np.unique(mask, return_counts=True)
    print("Original mask shape:", mask.shape)
    print("Original labels:", labels.tolist())

    print("\nLabel counts:")
    for label, count in zip(labels, counts):
        print(f"Label {int(label)}: {int(count):,} voxels")

    # Keep the original dataset labels unchanged.
    output_path = OUTPUT_DIR / "UPENN-GBM-00002_11_segm.nii.gz"

    output_img = nib.Nifti1Image(
        mask.astype(np.uint8),
        mask_img.affine,
        mask_img.header,
    )
    output_img.set_data_dtype(np.uint8)
    nib.save(output_img, str(output_path))

    # Reopen and verify the saved mask.
    saved_img = nib.load(str(output_path))
    saved_mask = np.asarray(saved_img.dataobj)

    if not np.array_equal(mask, saved_mask):
        raise ValueError("Saved mask differs from the original.")

    if not np.allclose(
        mask_img.affine, saved_img.affine, atol=1e-4
    ):
        raise ValueError("Saved mask affine differs from original.")

    print("\nSaved mask:", output_path)
    print("Saved mask verified successfully.")
    print("Original mask was not modified.")


if __name__ == "__main__":
    main()


from pathlib import Path

import nibabel as nib
import numpy as np

DATA_DIR = (
    Path.home()
    / "Documents"
    / "BrainTumorDataset"
    / "NIfTI"
)

MRI_FILES = [
    "UPENN-GBM-00002_11_FLAIR.nii",
    "UPENN-GBM-00002_11_T1.nii",
    "UPENN-GBM-00002_11_T1GD.nii",
    "UPENN-GBM-00002_11_T2.nii",
]

MASK_FILE = "UPENN-GBM-00002_11_segm.nii"


def main():
    mask_path = DATA_DIR / "Segmentation" / MASK_FILE

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask_img = nib.load(str(mask_path))
    mask = np.asarray(mask_img.dataobj)

    print("SUBJECT: UPENN-GBM-00002_11")
    print("Mask shape:", mask.shape)
    print("Mask orientation:", nib.aff2axcodes(mask_img.affine))

    voxel_volume_mm3 = abs(
        np.linalg.det(mask_img.affine[:3, :3])
    )

    print("\nMRI verification:")
    for filename in MRI_FILES:
        path = DATA_DIR / "MRI" / filename

        if not path.exists():
            print(f"MISSING: {filename}")
            continue

        image = nib.load(str(path))

        same_shape = image.shape == mask_img.shape
        same_affine = np.allclose(
            image.affine, mask_img.affine, atol=1e-4
        )

        print(f"\n{filename}")
        print("  Shape:", image.shape)
        print("  Dimensions match:", same_shape)
        print("  Affine matches:", same_affine)

    print("\nMask label volumes:")
    for label in np.unique(mask):
        count = int(np.count_nonzero(mask == label))
        volume_cm3 = count * voxel_volume_mm3 / 1000

        print(
            f"  Label {int(label)}: "
            f"{count:,} voxels, {volume_cm3:.3f} cm³"
        )


if __name__ == "__main__":
    main()

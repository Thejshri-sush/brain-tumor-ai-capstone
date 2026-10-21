
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

BASE = (
    Path.home()
    / "Documents"
    / "BrainTumorDataset"
    / "NIfTI"
)

MRI_PATH = BASE / "MRI" / "UPENN-GBM-00002_11_FLAIR.nii"
MASK_PATH = (
    BASE / "Segmentation" / "UPENN-GBM-00002_11_segm.nii"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "visualizations"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mri_img = nib.load(str(MRI_PATH))
mask_img = nib.load(str(MASK_PATH))

mri = mri_img.get_fdata()
mask = mask_img.get_fdata()

if mri.shape != mask.shape:
    raise ValueError("MRI and segmentation dimensions do not match.")

if not np.allclose(mri_img.affine, mask_img.affine, atol=1e-4):
    raise ValueError(
        "MRI and mask affine matrices differ. "
        "Check spatial alignment before using the mask."
    )

counts = np.count_nonzero(mask, axis=(0, 1))
slice_index = int(np.argmax(counts))

slice_data = mri[:, :, slice_index]
low, high = np.percentile(slice_data, [1, 99])

if high > low:
    slice_data = np.clip(
        (slice_data - low) / (high - low), 0, 1
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(slice_data.T, cmap="gray", origin="lower")
axes[0].set_title(f"FLAIR MRI — Slice {slice_index}")
axes[0].axis("off")

axes[1].imshow(slice_data.T, cmap="gray", origin="lower")
overlay = np.ma.masked_where(
    mask[:, :, slice_index].T == 0,
    mask[:, :, slice_index].T,
)
axes[1].imshow(
    overlay,
    cmap="jet",
    alpha=0.55,
    origin="lower",
    interpolation="nearest",
)
axes[1].set_title("FLAIR MRI + Tumour Mask")
axes[1].axis("off")

fig.tight_layout()

output_path = OUTPUT_DIR / "flair_tumour_overlay.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved visualization to:", output_path)
print("Selected slice:", slice_index)
print("Tumour-labelled voxels in volume:", int(np.count_nonzero(mask)))
print("Segmentation labels:", np.unique(mask).tolist())

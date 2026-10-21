
from pathlib import Path

import nibabel as nib
import numpy as np

DATA_DIR = (
    Path.home()
    / "Documents"
    / "BrainTumorDataset"
    / "NIfTI"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "processed"
)

SUBJECT = "UPENN-GBM-00002_11"

MRI_MODALITIES = ["FLAIR", "T1", "T1GD", "T2"]


def normalize_mri(image):
    """Z-score normalize nonzero MRI voxels."""
    image = image.astype(np.float32)
    brain_mask = np.isfinite(image) & (image != 0)

    normalized = np.zeros_like(image, dtype=np.float32)

    if not np.any(brain_mask):
        raise ValueError("MRI contains no nonzero voxels.")

    values = image[brain_mask]
    mean = values.mean()
    std = values.std()

    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("MRI intensity variation is too small.")

    normalized[brain_mask] = (values - mean) / std
    return normalized


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = (
        DATA_DIR / "MRI" / f"{SUBJECT}_FLAIR.nii"
    )
    reference_img = nib.load(str(reference_path))

    for modality in MRI_MODALITIES:
        input_path = DATA_DIR / "MRI" / f"{SUBJECT}_{modality}.nii"
        image_obj = nib.load(str(input_path))

        if image_obj.shape != reference_img.shape:
            raise ValueError(
                f"{modality}: dimensions differ from FLAIR."
            )

        if not np.allclose(
            image_obj.affine, reference_img.affine, atol=1e-4
        ):
            raise ValueError(
                f"{modality}: spatial alignment differs from FLAIR."
            )

        image = image_obj.get_fdata(dtype=np.float32)
        normalized = normalize_mri(image)

        output_path = (
            OUTPUT_DIR / f"{SUBJECT}_{modality}_normalized.nii.gz"
        )

        output_img = nib.Nifti1Image(
            normalized, image_obj.affine, image_obj.header
        )
        output_img.set_data_dtype(np.float32)
        nib.save(output_img, str(output_path))

        print(f"\nProcessed: {modality}")
        print("Output:", output_path)
        print("Shape:", normalized.shape)
        print("Normalized range:",
              float(normalized.min()), "to",
              float(normalized.max()))

    print("\nMRI preprocessing completed.")
    print("Original MRI files and tumour mask were not modified.")


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import exposure, filters, util


def preprocess_mri(
    input_path: str,
    output_path: str,
    image_size: tuple[int, int] = (256, 256),
) -> dict:
    """
    Prepare a 2D MRI image for later analysis.

    Steps:
    1. Load the image as grayscale.
    2. Resize it to a fixed size.
    3. Normalize pixel values to [0, 1].
    4. Apply light contrast enhancement.
    5. Save the processed image.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.is_file():
        raise FileNotFoundError(f"MRI image not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_file) as image:
        grayscale = image.convert("L")
        resized = grayscale.resize(image_size, Image.Resampling.LANCZOS)

    image_array = np.asarray(resized, dtype=np.float32) / 255.0

    # Enhance contrast without changing the image dimensions.
    enhanced = exposure.equalize_adapthist(image_array, clip_limit=0.02)

    # Save an 8-bit image for convenient inspection.
    output_array = util.img_as_ubyte(np.clip(enhanced, 0.0, 1.0))
    Image.fromarray(output_array).save(output_file)

    return {
        "input": str(input_file),
        "output": str(output_file),
        "shape": tuple(int(v) for v in enhanced.shape),
        "minimum": float(enhanced.min()),
        "maximum": float(enhanced.max()),
    }


if __name__ == "__main__":
    print("MRI preprocessing module loaded.")
    print("This module accepts a 2D image file when called with input and output paths.")

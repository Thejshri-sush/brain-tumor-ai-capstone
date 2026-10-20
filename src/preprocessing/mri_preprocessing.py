
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

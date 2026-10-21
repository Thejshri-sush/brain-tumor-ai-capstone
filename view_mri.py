from pathlib import Path
import pydicom
import numpy as np
import matplotlib.pyplot as plt

folder = (
    Path.home()
    / "Documents/BrainTumorDataset/DICOM/upenn_gbm/"
      "UPENN-GBM-00001/T2-FLAIR"
)

slices = []

for path in folder.rglob("*.dcm"):
    try:
        ds = pydicom.dcmread(path)

        if hasattr(ds, "PixelData"):
            image = ds.pixel_array.astype(np.float32)

            if image.ndim == 2:
                slices.append(image)

    except Exception as e:
        print("Skipped:", path.name, e)

if not slices:
    raise RuntimeError("No readable DICOM slices found.")

image = max(
    slices,
    key=lambda img: np.percentile(img, 99)
    - np.percentile(img, 1)
)

low, high = np.percentile(image, [1, 99])

plt.figure(figsize=(7, 7))
plt.imshow(image, cmap="gray", vmin=low, vmax=high)
plt.title("UPENN-GBM-00001 — T2-FLAIR MRI")
plt.axis("off")
plt.tight_layout()
plt.show()

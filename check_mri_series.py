from pathlib import Path
import pydicom

root = (
    Path.home()
    / "Documents/BrainTumorDataset/DICOM/upenn_gbm/UPENN-GBM-00001"
)

for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue

    files = list(folder.rglob("*.dcm"))
    if not files:
        continue

    ds = pydicom.dcmread(files[0], stop_before_pixels=True)

    print("\nSeries folder:", folder.name)
    print("Number of DICOM files:", len(files))
    print("Modality:", getattr(ds, "Modality", "Unknown"))
    print("Series description:", getattr(ds, "SeriesDescription", "Unknown"))
    print("Image dimensions:", getattr(ds, "Rows", "?"), "x",
          getattr(ds, "Columns", "?"))
    print("Series Instance UID:", getattr(ds, "SeriesInstanceUID", "Unknown"))

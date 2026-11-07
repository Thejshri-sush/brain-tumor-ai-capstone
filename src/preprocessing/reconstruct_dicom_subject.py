from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def reconstruct_subject(subject_id="UPENN-GBM-00002"):
    project_dir = Path(__file__).resolve().parents[2]

    dicom_dir = (
        Path.home()
        / "Documents/BrainTumorDataset/DICOM"
        / subject_id
        / "MRI_FLAIR"
    )

    output_dir = project_dir / "outputs" / "reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{subject_id}_FLAIR_dicom_reconstructed.nii.gz"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    files = sorted(dicom_dir.glob("*.dcm"))

    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    datasets = [pydicom.dcmread(path) for path in files]

    first = datasets[0]

    if len(datasets) != 60:
        raise ValueError(f"Expected 60 slices, found {len(datasets)}")

    # Verify series identity and image dimensions.
    for ds in datasets:
        if ds.SeriesInstanceUID != first.SeriesInstanceUID:
            raise ValueError("Multiple DICOM series detected.")

        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise ValueError("Inconsistent image dimensions detected.")

        if not np.allclose(ds.PixelSpacing, first.PixelSpacing, atol=1e-5):
            raise ValueError("Inconsistent pixel spacing detected.")

        if not np.allclose(
            ds.ImageOrientationPatient,
            first.ImageOrientationPatient,
            atol=1e-5,
        ):
            raise ValueError("Inconsistent image orientation detected.")

    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)

    # DICOM directions: columns first, then rows.
    column_direction = orientation[:3]
    row_direction = orientation[3:]

    normal = np.cross(column_direction, row_direction)
    normal /= np.linalg.norm(normal)

    # Sort by physical position, not filename.
    datasets.sort(
        key=lambda ds: np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=np.float64),
            normal,
        )
    )

    positions = np.asarray([
        np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=np.float64),
            normal,
        )
        for ds in datasets
    ])

    gaps = np.diff(positions)

    if np.any(gaps <= 0):
        raise ValueError("Duplicate or incorrectly ordered slice positions.")

    if not np.allclose(gaps, 3.0, atol=0.01):
        raise ValueError("Slice spacing is not consistently 3 mm.")

    # Reconstruct the complete 3D pixel volume.
    slices = []

    for ds in datasets:
        pixels = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        pixels = pixels * slope + intercept
        slices.append(pixels)

    # DICOM stack: (slices, rows, columns).
    volume = np.stack(slices, axis=0)

    # NIfTI axes: (columns, rows, slices).
    volume = np.transpose(volume, (2, 1, 0))

    # Construct the DICOM LPS affine.
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    slice_spacing = float(np.median(gaps))

    origin_lps = np.asarray(
        datasets[0].ImagePositionPatient,
        dtype=np.float64,
    )

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = column_direction * pixel_spacing[1]
    affine_lps[:3, 1] = row_direction * pixel_spacing[0]
    affine_lps[:3, 2] = normal * slice_spacing
    affine_lps[:3, 3] = origin_lps

    # Convert patient coordinates from LPS to RAS for NIfTI.
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    image = nib.Nifti1Image(volume, affine_ras)
    image.set_qform(affine_ras, code=1)
    image.set_sform(affine_ras, code=1)

    nib.save(image, output_path)

    print("DICOM reconstruction completed.")
    print("Subject:", subject_id)
    print("Slices:", len(datasets))
    print("Volume shape:", volume.shape)
    print("Data type:", volume.dtype)
    print("Voxel spacing (mm):", image.header.get_zooms()[:3])
    print("Intensity range:", float(volume.min()), "to", float(volume.max()))
    print("Affine (RAS):")
    print(affine_ras)
    print("Output:", output_path)


if __name__ == "__main__":
    reconstruct_subject()

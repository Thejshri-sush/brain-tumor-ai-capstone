from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.preprocessing.reconstruct_dicom_subject import reconstruct_subject


PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Documents/BrainTumorDataset/DICOM"


def process_subject(subject_id):
    subject_dir = DATA_ROOT / subject_id
    mri_dir = subject_dir / "MRI_FLAIR"
    seg_dir = subject_dir / "SEG"

    output_dir = PROJECT / "outputs"
    recon_dir = output_dir / "reconstructed"
    mask_dir = output_dir / "3d_masks_aligned_to_mri"
    visual_dir = output_dir / "visualizations"

    mri_path = recon_dir / f"{subject_id}_FLAIR_dicom_reconstructed.nii.gz"
    seg_path = recon_dir / f"{subject_id}_FLAIR_segmentation_labels.nii.gz"
    combined_path = output_dir / f"{subject_id}_combined_labels_full60.npy"
    overlay_path = visual_dir / f"{subject_id}_FLAIR_segmentation_overlay_check.png"

    names = {
        1: "Necrosis",
        2: "Edema",
        3: "Enhancing_Lesion",
    }

    mask_paths = {
        number: mask_dir / f"{subject_id}_{name}_mask.npy"
        for number, name in names.items()
    }

    destinations = [
        mri_path, seg_path, combined_path, overlay_path,
        *mask_paths.values()
    ]

    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n" + "\n".join(existing)
        )

    mri_files = list(mri_dir.glob("*.dcm"))
    seg_files = list(seg_dir.glob("*.dcm"))

    if len(mri_files) != 60:
        raise ValueError(f"Expected 60 MRI slices; found {len(mri_files)}")
    if len(seg_files) != 1:
        raise ValueError(f"Expected one SEG file; found {len(seg_files)}")

    print("\n===== STEP 1: RECONSTRUCT MRI =====")
    reconstruct_subject(subject_id)

    datasets = [pydicom.dcmread(path) for path in mri_files]
    first = datasets[0]

    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    col_dir = orientation[:3]
    row_dir = orientation[3:]
    normal = np.cross(col_dir, row_dir)
    normal /= np.linalg.norm(normal)

    datasets.sort(
        key=lambda ds: np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=float), normal
        )
    )

    positions = np.asarray([
        np.asarray(ds.ImagePositionPatient, dtype=float)
        for ds in datasets
    ])
    projections = positions @ normal

    if np.any(np.diff(projections) <= 0):
        raise ValueError("Duplicate or invalid MRI slice positions")

    image = nib.load(str(mri_path))
    volume = image.get_fdata(dtype=np.float32)

    if volume.shape != (first.Columns, first.Rows, 60):
        raise ValueError(f"Unexpected reconstructed MRI shape: {volume.shape}")

    print("\n===== STEP 2: VERIFY MRI PIXELS =====")
    for index, ds in enumerate(datasets):
        original = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        original = original * slope + intercept

        difference = np.max(np.abs(original.T - volume[:, :, index]))
        if difference != 0:
            raise ValueError(
                f"MRI pixel mismatch at slice {index}: {difference}"
            )

    print("60 slices checked; zero pixel mismatches.")

    print("\n===== STEP 3: READ SEG GEOMETRY =====")
    seg = pydicom.dcmread(seg_files[0])
    shared = seg.SharedFunctionalGroupsSequence[0]

    seg_orientation = np.asarray(
        shared.PlaneOrientationSequence[0].ImageOrientationPatient,
        dtype=float
    )
    seg_col = seg_orientation[:3]
    seg_row = seg_orientation[3:]

    col_dot = float(np.dot(col_dir, seg_col))
    row_dot = float(np.dot(row_dir, seg_row))

    if abs(col_dot) < 0.999 or abs(row_dot) < 0.999:
        raise ValueError("SEG orientation differs unexpectedly from MRI")

    col_flip = col_dot < 0
    row_flip = row_dot < 0

    if col_flip:
        raise ValueError("Unexpected column flip; manual review required")

    if not row_flip:
        raise ValueError("Expected row flip was not detected")

    mri_spacing = np.asarray(first.PixelSpacing, dtype=float)
    seg_spacing = np.asarray(
        shared.PixelMeasuresSequence[0].PixelSpacing, dtype=float
    )

    if not np.allclose(mri_spacing, seg_spacing, atol=1e-5):
        raise ValueError("MRI and SEG pixel spacing differ")

    if (seg.Rows, seg.Columns) != (first.Rows, first.Columns):
        raise ValueError("MRI and SEG dimensions differ")

    frame_pixels = seg.pixel_array
    frames = seg.PerFrameFunctionalGroupsSequence

    if len(frames) != int(seg.NumberOfFrames):
        raise ValueError("SEG frame count does not match metadata")

    masks = {
        number: np.zeros((60, first.Rows, first.Columns), dtype=np.uint8)
        for number in names
    }

    seen = set()
    slice_errors = []
    origin_errors = []
    empty_frames = 0

    print("\n===== STEP 4: MAP SEG FRAMES =====")

    for frame_index, frame in enumerate(frames):
        segment_number = int(
            frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        )

        if segment_number not in masks:
            raise ValueError(f"Unknown segment number: {segment_number}")

        pixels = np.asarray(frame_pixels[frame_index])

        if not np.isin(pixels, [0, 1]).all():
            raise ValueError(f"Non-binary pixels in frame {frame_index}")

        if not np.any(pixels):
            empty_frames += 1

        frame_position = np.asarray(
            frame.PlanePositionSequence[0].ImagePositionPatient,
            dtype=float
        )

        frame_projection = np.dot(frame_position, normal)
        slice_index = int(np.argmin(np.abs(projections - frame_projection)))

        mri_origin = positions[slice_index].copy()

        if row_flip:
            mri_origin += (
                (first.Rows - 1) * mri_spacing[0] * row_dir
            )

        difference = frame_position - mri_origin
        slice_error = abs(np.dot(difference, normal))
        inplane_difference = difference - np.dot(difference, normal) * normal
        origin_error = np.linalg.norm(inplane_difference)

        slice_errors.append(slice_error)
        origin_errors.append(origin_error)

        if slice_error >= 0.01 or origin_error >= 0.01:
            raise ValueError(
                f"Geometry mismatch in SEG frame {frame_index}: "
                f"slice={slice_error}, origin={origin_error}"
            )

        key = (segment_number, slice_index)
        if key in seen:
            raise ValueError(f"Duplicate segment/slice frame: {key}")
        seen.add(key)

        if row_flip:
            pixels = np.flipud(pixels)

        masks[segment_number][slice_index] = pixels.astype(np.uint8)

    print("Frames mapped:", len(frames))
    print("Unique MRI slices:", len(set(index for _, index in seen)))
    print("Row flip applied:", row_flip)
    print("Maximum slice error (mm):", max(slice_errors))
    print("Maximum in-plane origin error (mm):", max(origin_errors))
    print("Empty SEG frames:", empty_frames)

    print("\n===== STEP 5: CHECK MASKS =====")

    combined = np.zeros((60, first.Rows, first.Columns), dtype=np.uint8)

    for number, mask in masks.items():
        if np.any((combined > 0) & (mask > 0)):
            raise ValueError(f"Overlapping segmentation labels: {names[number]}")

        combined[mask > 0] = number

        print(
            names[number],
            "| Pixels:", int(mask.sum()),
            "| Slices:", int(np.count_nonzero(mask.sum(axis=(1, 2))))
        )

    print("Combined labels:", np.unique(combined))

    print("\n===== STEP 6: SAVE MASKS =====")
    mask_dir.mkdir(parents=True, exist_ok=True)

    for number, path in mask_paths.items():
        np.save(path, masks[number])

    np.save(combined_path, combined)

    labels_nifti = np.transpose(combined, (2, 1, 0)).astype(np.uint8)

    if labels_nifti.shape != image.shape:
        raise ValueError("Segmentation and MRI shapes differ")

    seg_image = nib.Nifti1Image(labels_nifti, image.affine)
    seg_image.set_data_dtype(np.uint8)
    seg_image.set_qform(image.affine, code=1)
    seg_image.set_sform(image.affine, code=1)
    nib.save(seg_image, str(seg_path))

    print("Saved segmentation NIfTI:", seg_path)

    print("\n===== STEP 7: VERIFY SAVED OUTPUTS =====")
    saved_seg = nib.load(str(seg_path))
    saved_labels = saved_seg.get_fdata().astype(np.uint8)

    if not np.array_equal(saved_labels, labels_nifti):
        raise ValueError("Saved segmentation voxels differ from source masks")

    if not np.allclose(saved_seg.affine, image.affine):
        raise ValueError("Saved segmentation affine differs from MRI")

    if not np.array_equal(np.load(combined_path), combined):
        raise ValueError("Saved combined mask differs from source")

    for number, path in mask_paths.items():
        if not np.array_equal(np.load(path), masks[number]):
            raise ValueError(f"Saved mask verification failed: {path.name}")

    print("All masks preserved.")
    print("Affine matches MRI.")
    print("Shape matches MRI.")
    print("Verification: PASSED")

    print("\n===== STEP 8: CREATE OVERLAY =====")
    visual_dir.mkdir(parents=True, exist_ok=True)

    nonzero = volume[volume > 0]
    vmin, vmax = np.percentile(nonzero, [2, 98])

    cmap = ListedColormap([
        (0, 0, 0, 0),
        (1, 0, 0, 0.45),
        (0, 1, 0, 0.45),
        (0, 0.4, 1, 0.55)
    ])

    active_slices = [
        index for index in range(60)
        if np.any(combined[index] > 0)
    ]

    if not active_slices:
        raise ValueError("No segmentation pixels found")

    columns = 6
    rows = int(np.ceil(len(active_slices) / columns))

    fig, axes = plt.subplots(
        rows, columns,
        figsize=(16, max(3, rows * 2.7)),
        squeeze=False
    )
    axes = axes.flatten()

    for panel, index in enumerate(active_slices):
        ax = axes[panel]
        ax.imshow(
            np.rot90(volume[:, :, index]),
            cmap="gray", vmin=vmin, vmax=vmax
        )
        overlay = np.ma.masked_where(
            np.rot90(combined[index].T) == 0,
            np.rot90(combined[index].T)
        )
        ax.imshow(
            overlay, cmap=cmap, vmin=0, vmax=3,
            interpolation="nearest"
        )
        ax.set_title(f"MRI slice {index}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(active_slices):]:
        ax.axis("off")

    fig.suptitle(
        f"{subject_id} | FLAIR Segmentation Overlay\n"
        "Red: Necrosis | Green: Edema | Blue: Enhancing Lesion"
    )
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Overlay:", overlay_path)
    print("\n===== PROCESSING COMPLETED =====")
    print(subject_id, "SUCCESS")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.preprocessing.process_dicom_subject "
            "UPENN-GBM-00006"
        )

    subject = sys.argv[1]

    if subject not in {
        f"UPENN-GBM-{number:05d}" for number in range(6, 11)
    }:
        raise SystemExit("Only Subjects 00006–00010 are allowed.")

    process_subject(subject)

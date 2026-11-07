# DICOM Dataset Validation Report

## Dataset
UPENN-GBM DICOM dataset

## Subjects
9 subjects with MRI FLAIR and corresponding SEG data:
UPENN-GBM-00002 to UPENN-GBM-00010.

## Validation Summary

| Validation | Result |
|---|---:|
| Total SEG frames | 398 |
| 3D position checks passed | 398 |
| 3D position checks failed | 0 |
| Missing MRI references | 0 |
| Non-binary masks | 0 |
| Invalid segment numbers | 0 |
| Empty masks | 1 |

## Segmentation Classes
1. Necrosis
2. Edema
3. Enhancing Lesion

## Empty Mask Observation
Subject UPENN-GBM-00009 contains one empty Necrosis
mask frame. Its MRI reference and physical position
passed validation. It is recorded separately and is
not automatically considered corrupted.

## Validation Conclusion
All 398 SEG frames passed the implemented 3D physical
position checks. MRI references, binary mask values,
and segment numbers passed the reported checks.

This validation confirms data consistency. It does not
establish clinical accuracy of the segmentation labels.

## Status
DICOM integrity validation completed.

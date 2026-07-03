"""Dosomics module — dose feature extraction, DTH/OVH computation."""
from typing import List, Tuple, Optional
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


def load_dose_image(file_paths: List[str]) -> sitk.Image:
    """Load RTDOSE DICOM series into a SimpleITK image with dose values in Gy.

    Applies DoseGridScaling from DICOM metadata to convert raw pixel values
    to absolute dose (Gy).
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_paths)
    dose_image = reader.Execute()

    # Apply DoseGridScaling from DICOM header
    first_ds = pydicom.dcmread(str(file_paths[0]), stop_before_pixels=True)
    scaling = float(getattr(first_ds, 'DoseGridScaling', 1.0))

    arr = sitk.GetArrayFromImage(dose_image).astype(np.float64)
    arr = arr * scaling

    result = sitk.GetImageFromArray(arr)
    result.SetSpacing(dose_image.GetSpacing())
    result.SetOrigin(dose_image.GetOrigin())
    if result.GetDimension() == dose_image.GetDimension():
        result.SetDirection(dose_image.GetDirection())

    return result


def resample_mask_to_image(mask: sitk.Image, target: sitk.Image) -> sitk.Image:
    """Resample a binary ROI mask to match target image geometry.

    Uses nearest-neighbor interpolation to preserve binary values.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def compute_dth(oar_mask_arr: np.ndarray, ptv_mask_arr: np.ndarray,
                spacing: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Distance-to-Target Histogram.

    For each OAR voxel, compute Euclidean distance to the nearest PTV voxel.
    Returns distance bins and fraction of OAR volume at each distance.

    Args:
        oar_mask_arr: 3D binary array (OAR region)
        ptv_mask_arr: 3D binary array (PTV/target region)
        spacing: voxel spacing (z, y, x) in mm

    Returns:
        (distance_bins, volume_fractions)
    """
    if ptv_mask_arr.sum() == 0 or oar_mask_arr.sum() == 0:
        return np.array([]), np.array([])

    # Distance transform: distance from each background voxel to nearest foreground
    # Invert PTV mask: True where PTV is NOT present
    inv_ptv = ~ptv_mask_arr.astype(bool)
    dist_map = distance_transform_edt(inv_ptv, sampling=spacing)

    # Get distances at OAR voxel locations
    oar_voxels = oar_mask_arr.astype(bool)
    distances = dist_map[oar_voxels]

    if len(distances) == 0:
        return np.array([]), np.array([])

    # Build histogram
    max_dist = float(np.percentile(distances, 99))
    bins = np.linspace(0, max_dist, 50)
    hist, edges = np.histogram(distances, bins=bins)
    fractions = hist / hist.sum()

    # Use bin centers
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, fractions


def compute_ovh(oar_mask_arr: np.ndarray, ptv_mask_arr: np.ndarray,
                spacing: Tuple[float, ...],
                max_dist: float = 50.0, step: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Overlap Volume Histogram.

    Expanding shells around PTV surface; count OAR volume fraction within each shell.

    Args:
        oar_mask_arr: 3D binary array (OAR region)
        ptv_mask_arr: 3D binary array (PTV/target region)
        spacing: voxel spacing (z, y, x) in mm
        max_dist: maximum distance from PTV surface (mm)
        step: shell thickness (mm)

    Returns:
        (distance_values, cumulative_overlap_fractions)
    """
    if ptv_mask_arr.sum() == 0 or oar_mask_arr.sum() == 0:
        return np.array([]), np.array([])

    oar_total = float(oar_mask_arr.sum())

    inv_ptv = ~ptv_mask_arr.astype(bool)
    dist_map = distance_transform_edt(inv_ptv, sampling=spacing)

    distances = np.arange(0, max_dist + step, step)
    overlaps = []

    for d in distances:
        # Shell: voxels within distance d of PTV
        shell = dist_map <= d
        overlap = np.logical_and(shell, oar_mask_arr.astype(bool))
        overlaps.append(float(overlap.sum()) / oar_total)

    return distances, np.array(overlaps)


def compute_dvh(dose_arr: np.ndarray, mask_arr: np.ndarray,
                max_dose: Optional[float] = None, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Dose-Volume Histogram for an ROI.

    Args:
        dose_arr: 3D dose array (Gy)
        mask_arr: 3D binary ROI mask
        max_dose: maximum dose for histogram (default: 99th percentile)
        n_bins: number of histogram bins

    Returns:
        (dose_bins, volume_fractions) — cumulative DVH
    """
    roi_dose = dose_arr[mask_arr.astype(bool)]
    if len(roi_dose) == 0:
        return np.array([]), np.array([])

    if max_dose is None:
        max_dose = float(np.percentile(roi_dose, 99.5))

    bins = np.linspace(0, max_dose, n_bins + 1)
    hist, _ = np.histogram(roi_dose, bins=bins)

    # Cumulative DVH: fraction of volume receiving >= dose
    cumulative = np.cumsum(hist[::-1])[::-1]
    cumulative = cumulative / cumulative[0] if cumulative[0] > 0 else cumulative

    dose_centers = (bins[:-1] + bins[1:]) / 2
    return dose_centers, cumulative


def extract_dose_features(dose_image: sitk.Image, masks_dict: dict,
                          feature_classes: Optional[dict] = None) -> 'pd.DataFrame':
    """Extract radiomics features from dose distribution for multiple ROIs.

    Reuses PyRadiomics engine with dose image as input.
    ROI masks are resampled to dose grid before extraction.

    Args:
        dose_image: SimpleITK dose image (Gy)
        masks_dict: {roi_name: mask_sitk_image}
        feature_classes: feature class config (default: all enabled)

    Returns:
        DataFrame with ROI names and dose features
    """
    import pandas as pd
    from src.feature_extractor import RadiomicsFeatureExtractor

    if feature_classes is None:
        feature_classes = {
            'firstorder': True,
            'shape': False,  # shape features don't make sense for dose
            'glcm': True,
            'gldm': True,
            'glrlm': True,
            'glszm': True,
            'ngtdm': True,
        }

    extractor = RadiomicsFeatureExtractor(feature_classes=feature_classes)
    rows = []

    for roi_name, mask in masks_dict.items():
        # Resample mask to dose grid
        resampled_mask = resample_mask_to_image(mask, dose_image)
        mask_arr = sitk.GetArrayFromImage(resampled_mask)
        if mask_arr.sum() == 0:
            continue

        features = extractor.extract_features(dose_image, resampled_mask)
        if features:
            row = {'ROI': roi_name}
            row.update(features)
            rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def find_ptv_rois(roi_names: List[str]) -> List[str]:
    """Identify PTV/target ROIs by name patterns.

    Matches common naming: PTV, CTV, GTV, Target, Prescription, Rx.
    """
    ptv_keywords = ['ptv', 'ctv', 'gtv', 'target', 'prescription', 'rx', 'plan']
    ptv_rois = []
    for name in roi_names:
        lower = name.lower()
        if any(kw in lower for kw in ptv_keywords):
            ptv_rois.append(name)
    return ptv_rois

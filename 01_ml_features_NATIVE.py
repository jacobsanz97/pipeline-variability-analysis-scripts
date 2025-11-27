#!/usr/bin/env python3
"""
NATIVE SPACE VERSION (V13: Native-Space Optimized) - AUTO-LABEL DETECTION
-------------------------------------------------------------------------------
This version processes segmentations in their native space.
Key Changes:
1. Updated PIPELINES to native space filenames (assumed).
2. load_seg uses SimpleITK to correctly read physical space information (spacing, affine).
3. Volume and Surface Area computation (sitk_morphological_stats_native) uses the image's native spacing.
4. PCA (fast_pca_on_physical_coords) is performed on physical coordinates (mm) derived from the image affine, ensuring physical accuracy for shape features.

Dependencies:
    pip install SimpleITK numpy pandas nibabel tqdm
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor 
import SimpleITK as sitk

# -----------------------------

# === OUTPUT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

# Dataset to exclude
EXCLUDED_DATASET = "FastSurferAnonBIDS_nipoppy"

# Pipelines and filenames: UPDATED TO NATIVE SPACE FILENAMES
PIPELINES = {
    "fslanat6071ants243": "T1_subcort_seg.nii.gz",
    "freesurfer741ants243": "aseg.nii.gz",
    "freesurfer8001ants243": "aseg.nii.gz", 
    "samseg8001ants243": "seg.nii.gz",
}

# Labels to exclude across all pipelines
EXCLUDED_LABELS = {258, 165, 72, 80, 62, 30, 77}

# MASTER LIST OF ALL AVAILABLE FEATURES
ALL_FEATURES = {
    'volume',
    'surface-area', 
    'sphericity',
    'compactness',
    'pca-eigenvalue-1',
    'pca-eigenvalue-2',
    'pca-eigenvalue-3',
}

# ------------------------------
# Fast utilities (Simplified)
# ------------------------------

def get_voxel_indices(mask, max_points=1000):
    """Returns voxel indices with intelligent downsampling for PCA."""
    coords = np.argwhere(mask) # Returns (D_idx, H_idx, W_idx) in numpy array order
    n_points = len(coords)
    
    if n_points == 0:
        return coords
    
    # For small regions, use all points
    if n_points <= max_points:
        return coords
    
    # For large regions, sample proportionally but cap at max_points
    step = max(1, n_points // max_points)
    return coords[::step]

# ------------------------------
# SimpleITK Feature Computation (NATIVE Optimized)
# ------------------------------

def sitk_morphological_stats_native(sitk_img_label):
    """
    Computes Volume and Surface Area using SimpleITK,
    relying on the image's native spacing and origin.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk_img_label)
    
    if stats.GetNumberOfLabels() == 0:
         return 0.0, 0.0

    # Volume Calculation (in mm^3)
    volume = stats.GetPhysicalSize(1)
    
    # Surface Area Calculation (in mm^2)
    boundary_img = sitk.BinaryContourImageFilter().Execute(sitk_img_label)
    boundary_filter = sitk.LabelShapeStatisticsImageFilter()
    boundary_filter.Execute(boundary_img) 
    surface_area = boundary_filter.GetPhysicalSize(1)

    return volume, surface_area

# ------------------------------
# OPTIMIZED PCA on Physical Coordinates
# ------------------------------

def fast_pca_on_physical_coords(sitk_img_label, coords_indices):
    """
    Highly optimized PCA computation for morphological features using physical coordinates (mm).
    Uses the image's native affine (via SimpleITK) to transform voxel indices.
    """
    n_points = len(coords_indices)
    
    # Early exits
    if n_points < 3:
        return [0.0, 0.0, 0.0]

    # Apply downsampling (the key optimization!)
    max_points = 1000
    if n_points > max_points:
        step = max(1, n_points // max_points)
        sampled_indices = coords_indices[::step]
        n_points = len(sampled_indices)
    else:
        sampled_indices = coords_indices
    
    # 1. Transform Voxel Indices to Physical Coordinates
    physical_coords = []
    
    for idx in sampled_indices:
        # NumPy argwhere returns indices in (D, H, W) order. 
        # SimpleITK indices are in (X, Y, Z) order, corresponding to (W, H, D).
        # We must reverse the NumPy indices (idx[2], idx[1], idx[0]) for SimpleITK.
        sitk_idx = [int(idx[2]), int(idx[1]), int(idx[0])] 
        
        physical_point = sitk_img_label.TransformIndexToPhysicalPoint(sitk_idx)
        physical_coords.append(physical_point)
        
    coords = np.array(physical_coords)
    n_points = len(coords)

    # Very small regions - use bounding box approximation
    if n_points < 10:
        ranges = coords.max(axis=0) - coords.min(axis=0)
        # Return squared ranges to approximate variance/eigenvalues
        return (ranges ** 2).tolist()  
    
    try:
        # Center coordinates
        centered = coords - coords.mean(axis=0)
        
        # Compute covariance matrix on physical coordinates
        if n_points < 5000:
            cov_matrix = np.cov(centered, rowvar=False, bias=True)
        else:
            # Manual covariance computation
            cov_matrix = (centered.T @ centered) / (n_points - 1)
        
        # Fast eigenvalue computation for symmetric matrices
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[::-1]  # Descending order
        
        # Eigenvalues are variance along the principal axes (in mm^2)
        return eigenvalues[:3].tolist()
        
    except (ValueError, np.linalg.LinAlgError, MemoryError):
        return [0.0, 0.0, 0.0]
    
# ------------------------------
# Combined feature computation
# ------------------------------

def compute_features_optimized(sitk_img_label, coords_indices, features_to_compute):
    """Compute features for a given mask in native space."""
    
    # Volume and Surface Area
    volume, surface = sitk_morphological_stats_native(sitk_img_label)

    f = {}
    
    # Primary Features
    if 'volume' in features_to_compute:
        f['volume'] = volume
    if 'surface-area' in features_to_compute:
        f['surface-area'] = surface

    # Derived Features
    has_vol_surf = volume > 0 and surface > 0
    
    if 'sphericity' in features_to_compute:
        if has_vol_surf:
            # Sphericity = pi^(1/3) * (6 * Volume)^(2/3) / Surface Area
            f['sphericity'] = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface
        else:
            f['sphericity'] = np.nan
            
    if 'compactness' in features_to_compute:
        if has_vol_surf:
            # Compactness = Volume^2 / Surface Area^3
            f['compactness'] = (volume ** 2) / (surface ** 3)
        else:
            f['compactness'] = np.nan

    # PCA Features (Physically accurate using native space affine)
    if any(k.startswith('pca-eigenvalue') for k in features_to_compute):
        # Pass the label's SimpleITK image for coordinate transformation
        p = fast_pca_on_physical_coords(sitk_img_label, coords_indices)
        pca_success = any(val != 0.0 for val in p)

        if 'pca-eigenvalue-1' in features_to_compute:
            f['pca-eigenvalue-1'] = p[0] if pca_success else np.nan
        if 'pca-eigenvalue-2' in features_to_compute:
            f['pca-eigenvalue-2'] = p[1] if pca_success else np.nan
        if 'pca-eigenvalue-3' in features_to_compute:
            f['pca-eigenvalue-3'] = p[2] if pca_success else np.nan

    return f

# ------------------------------
# Subject processing
# ------------------------------

def load_seg(path):
    """Load segmentation file, returning NumPy array and SimpleITK image with affine/spacing."""
    # Load with Nibabel for NumPy array
    img_nib = nib.load(path)
    data_np = img_nib.get_fdata(dtype=np.float32).astype(np.int16) 
    
    # Load with SimpleITK to preserve the affine/physical space information
    img_sitk = sitk.ReadImage(path)
    
    return data_np, img_sitk # Return both

def compute_subject(files, root_dir, features_to_compute, available_labels_per_pipeline):
    """The main feature computation function executed by the ProcessPool workers."""
    
    # 🌟 FIX: INITIALIZE 'row' HERE BEFORE ANY LOOPS 🌟
    dataset, sub, ses = extract_ids(files[0][0], root_dir)
    row = {"dataset": dataset, "subject": sub, "session": ses}
    # -----------------------------------------------

    # Compute features for all available labels in each pipeline
    for path, pipeline in files:
        # Load both the numpy array and the SimpleITK image
        seg_np, seg_sitk = load_seg(path) 
        available_labels = available_labels_per_pipeline[pipeline]

        for label in available_labels:
            if label == 0 or label in EXCLUDED_LABELS:
                continue
                
            mask_np = seg_np == label
            
            if mask_np.sum() == 0:
                continue

            # 1. Prepare SimpleITK image for the specific label (The SITK fix from before)
            sitk_img_label = seg_sitk == label # Use SITK thresholding to keep header info
            sitk_img_label = sitk.Cast(sitk_img_label, sitk.sitkInt8)

            # 2. PCA uses raw voxel indices from NumPy
            coords_indices = get_voxel_indices(mask_np)

            # 3. Compute features using the label-specific sitk image
            feats = compute_features_optimized(sitk_img_label, coords_indices, features_to_compute)
            
            # Maintain exact same column format: {pipeline}__label_{label_number}__{feature}
            for feature_name, value in feats.items():
                column_name = f"{pipeline}__label_{label}__{feature_name}"
                row[column_name] = value # This is where 'row' is used

    return row # 🌟 'row' is now defined and safe to return 🌟

# ------------------------------
# Process Pool Wrapper & File Helpers
# ------------------------------

def _subject_pool_wrapper(args):
    """Wrapper to unpack arguments for compute_subject to satisfy exe.map."""
    files, root_dir, features_to_compute, available_labels_per_pipeline = args
    return compute_subject(files, root_dir, features_to_compute, available_labels_per_pipeline)

def find_files(root, pipeline, filename):
    return sorted(glob(os.path.join(root, f"**/{pipeline}/**/{filename}"), recursive=True))

def extract_ids(path, rootdir):
    parts = Path(path).parts
    try:
        idx = parts.index(Path(rootdir).name)
        dataset = parts[idx+1]
    except ValueError:
        dataset = "unknown"
    sub = next((p for p in parts if p.startswith("sub-")), "nosub")
    ses = next((p for p in parts if p.startswith("ses-")), "nosess")
    return dataset, sub, ses

def get_available_labels_per_pipeline(complete_subjects, pipelines):
    """
    Pre-compute which labels actually exist in each pipeline by checking the first subject.
    Returns all non-zero labels found in the segmentations (excluding background and excluded labels).
    """
    available_labels_per_pipeline = {}
    
    print("Pre-computing available labels per pipeline...")
    print(f"Excluding labels: {sorted(EXCLUDED_LABELS)}")
    
    for pipeline in pipelines:
        # Find the first file for this pipeline
        first_file = None
        for files in complete_subjects.values():
            for file_path, file_pipeline in files:
                if file_pipeline == pipeline:
                    first_file = file_path
                    break
            if first_file:
                break
        
        if first_file:
            # Load the segmentation and find unique labels (only need the NumPy array here)
            seg_np, _ = load_seg(first_file) 
            unique_labels = set(np.unique(seg_np).astype(int))
            unique_labels.discard(0)  # Remove background
            # Remove excluded labels
            unique_labels = unique_labels - EXCLUDED_LABELS
            available_labels_per_pipeline[pipeline] = sorted(unique_labels)
            print(f"  {pipeline}: {len(unique_labels)} unique labels found: {list(unique_labels)}")
        else:
            available_labels_per_pipeline[pipeline] = []
            print(f"  {pipeline}: No files found")
    
    return available_labels_per_pipeline

# ------------------------------
# Main
# ------------------------------

def main(n_subjects, parallel, max_threads, selected_features):
    ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")

    print("=== AUTO-LABEL MORPHOLOGICAL FEATURE COMPUTATION (V13: Native-Space) ===")
    print(f"Core Optimizations: Native-space affine, SimpleITK volume/surface, Physical PCA")
    print(f"Features Selected: {', '.join(selected_features)}")
    print(f"Excluded labels: {sorted(EXCLUDED_LABELS)}")
    
    print("\nGathering files…")

    subj_map = defaultdict(list)
    for pl, fname in PIPELINES.items():
        fs = find_files(ROOT_DIR, pl, fname)
        print(f"  {pl}: {len(fs)} files")
        for f in fs:
            dataset, sub, ses = extract_ids(f, ROOT_DIR)
            if dataset == EXCLUDED_DATASET:
                continue
            sid = f"{dataset}__{sub}__{ses}"
            subj_map[sid].append((f, pl))

    complete = {sid:fls for sid,fls in subj_map.items() if set(pl for _,pl in fls)==set(PIPELINES)}
    print(f"Subjects with all pipelines: {len(complete)}")

    # Pre-compute available labels per pipeline
    available_labels_per_pipeline = get_available_labels_per_pipeline(complete, PIPELINES.keys())

    items = list(complete.items())
    if n_subjects > 0:
        items = items[:n_subjects]
        print(f"Limiting to {n_subjects} subjects")

    # The iterable for ProcessPoolExecutor
    iterable_for_pool = [
        (files, ROOT_DIR, selected_features, available_labels_per_pipeline) 
        for sid, files in items
    ]
    
    results = []

    if parallel:
        with ProcessPoolExecutor(max_workers=max_threads) as exe:
            for r in tqdm(exe.map(_subject_pool_wrapper, iterable_for_pool), total=len(iterable_for_pool)):
                results.append(r)
    else:
        # Serial execution
        for args in tqdm(iterable_for_pool):
            results.append(compute_subject(*args))

    df = pd.DataFrame(results)
    
    # Remove completely empty columns (where all values are NaN)
    initial_cols = len(df.columns)
    df = df.dropna(axis=1, how='all')
    final_cols = len(df.columns)
    
    print(f"Removed {initial_cols - final_cols} empty columns")
    
    # Reorganize columns: ID columns first, then feature columns
    id_cols = ["dataset", "subject", "session"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    
    df = df[id_cols + sorted(feature_cols)]

    out = Path(OUTPUT_DIR)/"morphological_features_native.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    print(f"Final dataset: {len(df)} subjects × {len(df.columns)} columns")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Auto-label morphological feature computation")
    p.add_argument("--n_subjects", type=int, default=0, help="Limit processing to the first N subjects.")
    p.add_argument("--parallel", action="store_true", help="Enable parallel processing using ProcessPoolExecutor.")
    p.add_argument("--max_threads", type=int, default=10, help="Maximum number of processes (workers) to use in parallel.")
    p.add_argument(
        "--features", 
        nargs='+', 
        default=list(ALL_FEATURES), 
        choices=ALL_FEATURES,
        help="List of features to compute (space-separated). Defaults to all features."
    )
    a = p.parse_args()
    
    main(a.n_subjects, a.parallel, a.max_threads, set(a.features))

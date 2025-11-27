#!/usr/bin/env python3
"""
MAXIMUM SPEED VERSION (V12: MNI-Space Optimized) - AUTO-LABEL DETECTION
-------------------------------------------------------------------------------
This version automatically detects all available labels in each pipeline's 
segmentation files and computes features for every label found.

Key Changes:
1. Removed hardcoded structure labels
2. Auto-detects labels from first segmentation file per pipeline  
3. Computes features for all non-zero labels
4. Maintains exact same output format: {pipeline}__label_{label_number}__{feature}
5. Excludes specific labels: 258, 165, 72, 80, 62, 30, 77

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

# Pipelines and filenames
PIPELINES = {
    "freesurfer741ants243": "aseg_MNI.nii.gz",
    "freesurfer8001ants243": "aseg_MNI.nii.gz", 
    "fslanat6071ants243": "subcortical_seg_MNI_ANTs.nii.gz",
    "samseg8001ants243": "seg_MNI.nii.gz",
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
    coords = np.argwhere(mask)
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
# SimpleITK Feature Computation (MNI Optimized)
# ------------------------------

def sitk_morphological_stats_mni(mask_np):
    """
    Computes Volume and Surface Area using SimpleITK, assuming MNI 1mm^3 isotropic spacing.
    """
    if mask_np.sum() == 0:
        return 0.0, 0.0
    
    # Convert NumPy mask to SimpleITK Image
    sitk_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    sitk_img.SetSpacing((1.0, 1.0, 1.0)) 
    
    # Volume Calculation
    volume_filter = sitk.LabelShapeStatisticsImageFilter()
    volume_filter.Execute(sitk_img)
    volume = volume_filter.GetPhysicalSize(1)
    
    # Surface Area Calculation
    boundary_img = sitk.BinaryContourImageFilter().Execute(sitk_img)
    boundary_filter = sitk.LabelShapeStatisticsImageFilter()
    boundary_filter.Execute(boundary_img) 
    surface_area = boundary_filter.GetPhysicalSize(1)

    return volume, surface_area

# ------------------------------
# OPTIMIZED PCA (NumPy Linear Algebra)
# ------------------------------

def fast_pca_from_indices(coords_indices):
    """Highly optimized PCA computation for morphological features."""
    coords = coords_indices
    n_points = len(coords)
    
    # Early exits
    if n_points < 3:
        return [0.0, 0.0, 0.0]
    
    # Apply downsampling if we have too many points (the key optimization!)
    max_points = 1000
    if n_points > max_points:
        step = max(1, n_points // max_points)
        coords = coords[::step]
        n_points = len(coords)
    
    # Very small regions - use bounding box approximation
    if n_points < 10:
        ranges = coords.max(axis=0) - coords.min(axis=0)
        return (ranges ** 2).tolist()  # Square to approximate variance
    
    try:
        # Center coordinates
        centered = coords - coords.mean(axis=0)
        
        # Choose covariance computation method based on size
        if n_points < 5000:
            cov_matrix = np.cov(centered, rowvar=False, bias=True)
        else:
            cov_matrix = (centered.T @ centered) / (n_points - 1)
        
        # Fast eigenvalue computation for symmetric matrices
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[::-1]  # Descending order
        
        return eigenvalues[:3].tolist()
        
    except (ValueError, np.linalg.LinAlgError, MemoryError):
        return [0.0, 0.0, 0.0]
# ------------------------------
# Combined feature computation
# ------------------------------

def compute_features_optimized(mask_np, coords_indices, features_to_compute):
    """Compute features for a given mask."""
    volume, surface = sitk_morphological_stats_mni(mask_np)

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
            f['sphericity'] = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface
        else:
            f['sphericity'] = np.nan
            
    if 'compactness' in features_to_compute:
        if has_vol_surf:
            f['compactness'] = (volume ** 2) / (surface ** 3)
        else:
            f['compactness'] = np.nan

    # PCA Features
    if any(k.startswith('pca-eigenvalue') for k in features_to_compute):
        p = fast_pca_from_indices(coords_indices)
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
    """Load segmentation file."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32).astype(np.int16) 
    return data

def compute_subject(files, root_dir, features_to_compute, available_labels_per_pipeline):
    """The main feature computation function executed by the ProcessPool workers."""
    dataset, sub, ses = extract_ids(files[0][0], root_dir)
    row = {"dataset": dataset, "subject": sub, "session": ses}

    # Compute features for all available labels in each pipeline
    for path, pipeline in files:
        seg = load_seg(path)
        available_labels = available_labels_per_pipeline[pipeline]

        for label in available_labels:
            if label == 0 or label in EXCLUDED_LABELS:  # Skip background and excluded labels
                continue
                
            mask_np = seg == label
            
            if mask_np.sum() == 0:
                continue

            # PCA uses raw voxel indices
            coords_indices = get_voxel_indices(mask_np)

            feats = compute_features_optimized(mask_np, coords_indices, features_to_compute)
            
            # Maintain exact same column format: {pipeline}__label_{label_number}__{feature}
            for feature_name, value in feats.items():
                column_name = f"{pipeline}__label_{label}__{feature_name}"
                row[column_name] = value

    return row

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
            # Load the segmentation and find unique labels (excluding background and excluded labels)
            seg = load_seg(first_file)
            unique_labels = set(np.unique(seg).astype(int))
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

    print("=== AUTO-LABEL MORPHOLOGICAL FEATURE COMPUTATION (V12: MNI-Space) ===")
    print(f"Core Optimizations: Auto-label detection, SimpleITK, Direct NumPy PCA")
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

    out = Path(OUTPUT_DIR)/"morphological_features_mni.csv"
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

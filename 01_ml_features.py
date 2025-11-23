#!/usr/bin/env python3
"""
MAXIMUM SPEED VERSION (V12: MNI-Space Optimized)
-------------------------------------------------------------------------------
This version assumes all input segmentations are reliably registered to MNI-space
with 1mm^3 isotropic voxels. It removes all complex affine/spacing handling and
relies solely on raw voxel counts and indices for maximum speed and simplicity.

Key Changes:
1. Affine Removal: Functions for spacing and world coordinates are removed/simplified.
2. Feature Calculation: Uses mask sum (voxel count) and SimpleITK with a fixed 1.0 spacing.
3. PCA: Calculates directly from raw voxel indices (np.argwhere) which is sufficient
   in MNI-aligned space.

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
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor 
import SimpleITK as sitk
# -----------------------------

# === OUTPUT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

# Dataset to exclude
EXCLUDED_DATASET = "FastSurferAnonBIDS_nipoppy"

# Pipelines and filenames
PIPELINES = {
    "freesurfer741ants243": "aseg_MNI.nii.gz", # Assuming MNI filenames
    "freesurfer8001ants243": "aseg_MNI.nii.gz",
    "fslanat6071ants243": "subcortical_seg_MNI_ANTs.nii.gz",
    "samseg8001ants243": "seg_MNI.nii.gz",
}

# Structures
STRUCTURE_LABELS = OrderedDict([
    ("Left-Thalamus-Proper", 10),
    ("Left-Caudate", 11),
    ("Left-Putamen", 12),
    ("Left-Pallidum", 13),
    ("Brain-Stem/4thVentricle", 16),
    ("Left-Hippocampus", 17),
    ("Left-Amygdala", 18),
    ("Left-Accumbens-area", 26),
    ("Right-Thalamus-Proper", 49),
    ("Right-Caudate", 50),
    ("Right-Putamen", 51),
    ("Right-Pallidum", 52),
    ("Right-Hippocampus", 53),
    ("Right-Amygdala", 54),
    ("Right-Accumbens-area", 58),
])

# MASTER LIST OF ALL AVAILABLE FEATURES (Clean API names maintained)
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

def get_voxel_indices(mask, step=2):
    """
    Returns raw voxel indices (i, j, k).
    In MNI space, these indices are sufficient for PCA.
    """
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return coords
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
    
    # 1. Convert NumPy mask to SimpleITK Image
    sitk_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    # CRITICAL: Fix spacing to 1.0, assuming MNI space
    sitk_img.SetSpacing((1.0, 1.0, 1.0)) 
    
    # --- Volume Calculation (Standard) ---
    volume_filter = sitk.LabelShapeStatisticsImageFilter()
    volume_filter.Execute(sitk_img)
    # Physical Size = Voxel Count * 1.0
    volume = volume_filter.GetPhysicalSize(1)
    
    # --- Surface Area Calculation (Robust two-step) ---
    boundary_img = sitk.BinaryContourImageFilter().Execute(sitk_img)
    boundary_filter = sitk.LabelShapeStatisticsImageFilter()
    boundary_filter.Execute(boundary_img) 
    # Physical Size = Boundary Voxel Count * 1.0
    surface_area = boundary_filter.GetPhysicalSize(1)

    return volume, surface_area

# ------------------------------
# OPTIMIZED PCA (NumPy Linear Algebra)
# ------------------------------

def fast_pca_from_indices(coords_indices):
    """
    Computes PCA eigenvalues directly from raw voxel indices (i, j, k).
    This is valid because MNI registration handles rotation/alignment.
    """
    coords = coords_indices
    if len(coords) < 3:
        return [0.0, 0.0, 0.0]

    try:
        # np.cov uses the index coordinates. Since MNI is aligned, this works.
        cov_matrix = np.cov(coords, rowvar=False) 
    except ValueError:
        return [0.0, 0.0, 0.0]

    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[::-1] # Reverse to get largest first

    if len(eigenvalues) < 3:
        return np.pad(eigenvalues, (0, 3 - len(eigenvalues)), constant_values=0).tolist()
    
    return eigenvalues[:3].tolist()

# ------------------------------
# Combined feature computation
# ------------------------------

def compute_features_optimized(mask_np, coords_indices, features_to_compute):
    
    # SimpleITK is run assuming 1.0 isotropic spacing
    volume, surface = sitk_morphological_stats_mni(mask_np)

    f = {}
    
    # --- Primary Features: Always populated (as voxel counts/boundary counts) ---
    if 'volume' in features_to_compute:
        f['volume'] = volume
    if 'surface-area' in features_to_compute:
        f['surface-area'] = surface

    # --- Derived Features ---
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

    # --- PCA Features ---
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
    # We still need nibabel to load the file, but we no longer use the affine
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32).astype(np.int16) 
    return data

def compute_subject(files, root_dir, features_to_compute):
    """The main feature computation function executed by the ProcessPool workers."""
    dataset, sub, ses = extract_ids(files[0][0], root_dir)
    row = {"dataset": dataset, "subject": sub, "session": ses}

    # Generate a list of all possible output columns based on selected features
    all_output_cols = [
        f"{pipeline}__{struct}__{k}" 
        for pipeline in PIPELINES 
        for struct in STRUCTURE_LABELS 
        for k in features_to_compute
    ]
    for col in all_output_cols:
        row[col] = np.nan

    for path, pipeline in files:
        seg = load_seg(path)

        for struct, label in STRUCTURE_LABELS.items():
            mask_np = seg == label
            
            if mask_np.sum() == 0:
                continue

            # PCA now uses raw voxel indices
            coords_indices = get_voxel_indices(mask_np)

            feats = compute_features_optimized(mask_np, coords_indices, features_to_compute)
            
            for k,v in feats.items():
                row[f"{pipeline}__{struct}__{k}"] = v

    return row

# ------------------------------
# Process Pool Wrapper & File Helpers (Unchanged)
# ------------------------------

def _subject_pool_wrapper(args):
    """Wrapper to unpack arguments for compute_subject to satisfy exe.map."""
    files, root_dir, features_to_compute = args
    return compute_subject(files, root_dir, features_to_compute)

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

# ------------------------------
# Main (Unchanged)
# ------------------------------

def main(n_subjects, parallel, max_threads, selected_features):
    ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")

    print("=== MAXIMUM SPEED MORPHOLOGICAL FEATURE COMPUTATION (V12: MNI-Space) ===")
    print(f"Core Optimizations: SimpleITK (Fixed 1mm Spacing), Direct NumPy PCA on indices.")
    print(f"Features Selected: {', '.join(selected_features)}")
    print("Gathering files…")

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

    items = list(complete.items())
    if n_subjects > 0:
        items = items[:n_subjects]
        print(f"Limiting to {n_subjects} subjects")

    # The iterable for ProcessPoolExecutor must be a list of tuples: (files, root_dir, features)
    iterable_for_pool = [(files, ROOT_DIR, selected_features) for sid, files in items]
    
    results = []

    if parallel:
        with ProcessPoolExecutor(max_workers=max_threads) as exe:
            for r in tqdm(exe.map(_subject_pool_wrapper, iterable_for_pool), total=len(iterable_for_pool)):
                results.append(r)
    else:
        # Serial execution
        for files, root_dir, features_to_compute in tqdm(iterable_for_pool):
            results.append(compute_subject(files, root_dir, features_to_compute))

    df = pd.DataFrame(results)
    id_cols = ["dataset","subject","session"]
    # Filter columns to only include the requested features
    feature_cols = sorted([
        c for c in df.columns 
        if c not in id_cols and any(f in c for f in selected_features)
    ])
    
    df = df[id_cols + feature_cols]

    out = Path(OUTPUT_DIR)/"morphological_features_mni.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="High-speed morphological feature computation (SimpleITK-accelerated)")
    p.add_argument("--n_subjects", type=int, default=0, help="Limit processing to the first N subjects.")
    p.add_argument("--parallel", action="store_true", help="Enable parallel processing using ProcessPoolExecutor.")
    p.add_argument("--max_threads", type=int, default=10, help="Maximum number of processes (workers) to use in parallel.")
    # API Modification: Allow selection of features
    p.add_argument(
        "--features", 
        nargs='+', 
        default=list(ALL_FEATURES), 
        choices=ALL_FEATURES,
        help="List of features to compute (space-separated). Defaults to all features."
    )
    a = p.parse_args()
    
    main(a.n_subjects, a.parallel, a.max_threads, set(a.features))

#!/usr/bin/env python3
"""
Compute morphological features for subcortical segmentations across multiple pipelines.
Features: volume, surface area, sphericity, compactness, convexity, PCA features.
One row per subject with features for all pipelines.
Outputs both raw and Z-score normalized versions.
Excludes FastSurferAnonBIDS_nipoppy dataset.
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
from nibabel.processing import resample_from_to
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

# === OUTPUT IN CURRENT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

# Dataset to exclude
EXCLUDED_DATASET = "FastSurferAnonBIDS_nipoppy"

PIPELINES = {
    "freesurfer741ants243": "aseg.nii.gz",
    "freesurfer8001ants243": "aseg.nii.gz", 
    "fslanat6071ants243": "T1_subcort_seg.nii.gz",
    "samseg8001ants243": "seg.nii.gz",
}

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

# Features that need Z-score normalization
FEATURES_TO_NORMALIZE = [
    'volume', 'surface-area', 
    'pca-eigenvalue-1', 'pca-eigenvalue-2', 'pca-eigenvalue-3'
]

# Features that don't need normalization (already normalized)
NON_NORMALIZED_FEATURES = [
    'sphericity', 'compactness', 'convexity',
    'pca-direction-1', 'pca-direction-2'
]

def find_files(root, pipeline, filename):
    """Find segmentation files recursively."""
    return sorted(glob(os.path.join(root, f"**/{pipeline}/**/{filename}"), recursive=True))

def extract_ids(path, root_dir):
    """Extract dataset, subject, and session from file path."""
    parts = Path(path).parts
    try:
        root_index = parts.index(Path(root_dir).name)
        dataset = parts[root_index + 1]
    except ValueError:
        dataset = "unknown"
    sub = next((p for p in parts if p.startswith("sub-")), "nosub")
    ses = next((p for p in parts if p.startswith("ses-")), "nosess")
    return dataset, sub, ses

def compute_surface_area(mask, voxel_volume):
    """Compute surface area using marching cubes approximation."""
    # Use binary erosion to find surface voxels
    eroded = ndimage.binary_erosion(mask)
    surface_voxels = np.logical_xor(mask, eroded)
    return surface_voxels.sum() * voxel_volume ** (2/3)  # Surface area scaling

def compute_sphericity(volume, surface_area):
    """Compute sphericity: how spherical the object is (1 = perfect sphere)."""
    if volume <= 0 or surface_area <= 0:
        return 0.0
    return (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area

def compute_compactness(volume, surface_area):
    """Compute compactness: (volume^2) / (surface_area^3)."""
    if volume <= 0 or surface_area <= 0:
        return 0.0
    return (volume ** 2) / (surface_area ** 3)

def compute_convexity(mask, voxel_volume):
    """Compute convexity: ratio of volume to convex hull volume."""
    if mask.sum() == 0:
        return 0.0
    
    # Get coordinates of mask voxels
    coords = np.argwhere(mask)
    
    if len(coords) < 4:  # Need at least 4 points for 3D convex hull
        return 1.0  # Assume convex for very small structures
    
    try:
        # Compute convex hull
        hull = ConvexHull(coords)
        convex_hull_volume = hull.volume * voxel_volume
        volume = mask.sum() * voxel_volume
        
        if convex_hull_volume <= 0:
            return 1.0
            
        return volume / convex_hull_volume
    except:
        # If convex hull fails, return 1.0 (assume convex)
        return 1.0

def compute_pca_features(mask, voxel_dims):
    """Compute PCA-based features: eigenvalues and eigenvectors."""
    coords = np.argwhere(mask)
    if len(coords) < 3:
        return [0, 0, 0, 0, 0]  # Not enough points for PCA
    
    # Convert to physical coordinates
    physical_coords = coords * voxel_dims
    
    pca = PCA()
    pca.fit(physical_coords)
    
    # Return eigenvalues (variance explained) and first principal direction
    eigenvalues = pca.explained_variance_
    # Pad if needed
    if len(eigenvalues) < 3:
        eigenvalues = np.pad(eigenvalues, (0, 3 - len(eigenvalues)))
    
    return eigenvalues[:3].tolist() + pca.components_[0][:2].tolist()

def compute_structure_features(mask, voxel_dims, voxel_volume):
    """Compute all features for a single structure."""
    features = {}
    
    # Basic volume
    volume = mask.sum() * voxel_volume
    features['volume'] = volume
    
    # Surface area
    surface_area = compute_surface_area(mask, voxel_volume)
    features['surface-area'] = surface_area
    
    # Sphericity
    features['sphericity'] = compute_sphericity(volume, surface_area)
    
    # Compactness
    features['compactness'] = compute_compactness(volume, surface_area)
    
    # Convexity
    features['convexity'] = compute_convexity(mask, voxel_volume)
    
    # PCA features (5 features: 3 eigenvalues + 2 direction components)
    pca_features = compute_pca_features(mask, voxel_dims)
    features['pca-eigenvalue-1'] = pca_features[0]
    features['pca-eigenvalue-2'] = pca_features[1]
    features['pca-eigenvalue-3'] = pca_features[2]
    features['pca-direction-1'] = pca_features[3]
    features['pca-direction-2'] = pca_features[4]
    
    return features

def load_segmentation_as_int(path):
    """Load segmentation file."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32).astype(np.int16)
    return data, img

def compute_features_for_subject(subject_files, root_dir):
    """Compute features for a single subject across all pipelines."""
    # Extract subject info from first file
    dataset, subject, session = extract_ids(subject_files[0][0], root_dir)
    
    features_row = {
        "dataset": dataset,
        "subject": subject, 
        "session": session
    }
    
    # Process each pipeline for this subject
    for file_path, pipeline in subject_files:
        try:
            seg_data, seg_img = load_segmentation_as_int(file_path)
            voxel_dims = np.sqrt(np.sum(seg_img.affine[:3, :3] ** 2, axis=0))  # Voxel dimensions in mm
            voxel_volume = np.prod(voxel_dims)  # Voxel volume in mm³
            
            # Process each structure for this pipeline
            for structure_name, label in STRUCTURE_LABELS.items():
                mask = seg_data == label
                
                if mask.sum() == 0:
                    # Structure not found, fill with zeros/NaNs
                    for feature in FEATURES_TO_NORMALIZE + NON_NORMALIZED_FEATURES:
                        col_name = f"{pipeline}__{structure_name}__{feature}"
                        if feature in ['pca-direction-1', 'pca-direction-2']:
                            features_row[col_name] = 0.0
                        else:
                            features_row[col_name] = np.nan
                else:
                    # Compute features
                    structure_features = compute_structure_features(mask, voxel_dims, voxel_volume)
                    
                    # Add to results with pipeline-structure-feature naming
                    for feature_name, value in structure_features.items():
                        col_name = f"{pipeline}__{structure_name}__{feature_name}"
                        features_row[col_name] = value
                        
        except Exception as e:
            print(f"Error processing {pipeline} for {subject}: {e}")
            # Fill with NaNs for this pipeline
            for structure_name in STRUCTURE_LABELS:
                for feature in FEATURES_TO_NORMALIZE + NON_NORMALIZED_FEATURES:
                    col_name = f"{pipeline}__{structure_name}__{feature}"
                    if feature in ['pca-direction-1', 'pca-direction-2']:
                        features_row[col_name] = 0.0
                    else:
                        features_row[col_name] = np.nan
    
    return features_row

def zscore_normalize_features(df):
    """Apply Z-score normalization to features that need it."""
    print("Applying Z-score normalization...")
    
    # Create a copy for normalized data
    df_norm = df.copy()
    
    # Get all feature columns (exclude ID columns)
    feature_cols = [col for col in df.columns if col not in ['dataset', 'subject', 'session']]
    
    # Identify columns that need normalization
    cols_to_normalize = []
    for col in feature_cols:
        # Check if this column contains any of the features that need normalization
        for feature in FEATURES_TO_NORMALIZE:
            if f"__{feature}" in col:
                cols_to_normalize.append(col)
                break
    
    print(f"Normalizing {len(cols_to_normalize)} feature columns...")
    
    # Apply Z-score normalization
    scaler = StandardScaler()
    df_norm[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    return df_norm

def main(n_subjects=0, parallel=False, max_threads=10):
    ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")
    
    print("=== Morphological Feature Computation ===")
    print(f"Excluding dataset: {EXCLUDED_DATASET}")
    print("Features: volume, surface-area, sphericity, compactness, convexity, PCA")
    print("Gathering segmentation files...")
    
    # Gather all files across all pipelines and group by subject
    subject_files_map = defaultdict(list)
    
    for pl, fname in PIPELINES.items():
        files = find_files(ROOT_DIR, pl, fname)
        print(f"  {pl}: {len(files)} files")
        
        for file_path in files:
            dataset, subject, session = extract_ids(file_path, ROOT_DIR)
            
            # Skip excluded dataset
            if dataset == EXCLUDED_DATASET:
                continue
                
            subject_id = f"{dataset}__{subject}__{session}"
            subject_files_map[subject_id].append((file_path, pl))
    
    # Filter subjects that have all pipelines
    complete_subjects = {}
    excluded_count = 0
    
    for subject_id, files in subject_files_map.items():
        pipelines_found = set(pl for _, pl in files)
        if pipelines_found == set(PIPELINES.keys()):
            # Double-check that dataset is not excluded
            dataset = subject_id.split("__")[0]
            if dataset != EXCLUDED_DATASET:
                complete_subjects[subject_id] = files
            else:
                excluded_count += 1
    
    print(f"Total subjects with all {len(PIPELINES)} pipelines: {len(complete_subjects)}")
    if excluded_count > 0:
        print(f"Excluded {excluded_count} subjects from {EXCLUDED_DATASET}")
    
    # Limit subjects if requested
    subject_items = list(complete_subjects.items())
    if n_subjects > 0:
        subject_items = subject_items[:n_subjects]
        print(f"Limited to {n_subjects} subjects")
    
    if len(subject_items) == 0:
        print("No subjects found after filtering! Check your data paths and pipeline definitions.")
        return
    
    results = []
    
    # Function to compute features for one subject
    def compute_for_subject(subject_info):
        subject_id, files = subject_info
        return compute_features_for_subject(files, ROOT_DIR)

    if parallel:
        with ThreadPoolExecutor(max_workers=max_threads) as exe:
            for r in tqdm(exe.map(compute_for_subject, subject_items), total=len(subject_items),
                          desc="Feature computation"):
                results.append(r)
    else:
        for subject_info in tqdm(subject_items, desc="Feature computation"):
            results.append(compute_for_subject(subject_info))

    # Create DataFrame and save raw features
    print("\nCreating output DataFrames...")
    df_raw = pd.DataFrame(results)
    
    # Reorder columns: dataset, subject, session first, then alphabetical
    cols = df_raw.columns.tolist()
    id_cols = ["dataset", "subject", "session"]
    other_cols = sorted([c for c in cols if c not in id_cols])
    df_raw = df_raw[id_cols + other_cols]
    
    # Create normalized version
    df_norm = zscore_normalize_features(df_raw)
    
    # Save both versions
    out_csv_raw = Path(OUTPUT_DIR) / "morphological_features_raw.csv"
    out_csv_norm = Path(OUTPUT_DIR) / "morphological_features_zscore.csv"
    
    df_raw.to_csv(out_csv_raw, index=False)
    df_norm.to_csv(out_csv_norm, index=False)
    
    print(f"Saved {out_csv_raw}")
    print(f"Saved {out_csv_norm}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total subjects processed: {len(df_raw)}")
    print(f"Total features per subject: {len(df_raw.columns) - 3}")  # minus ID columns
    print(f"Included datasets: {df_raw['dataset'].unique().tolist()}")
    
    # Show feature breakdown
    feature_cols = [c for c in df_raw.columns if c not in ['dataset', 'subject', 'session']]
    normalized_count = sum(1 for col in feature_cols if any(f"__{f}" in col for f in FEATURES_TO_NORMALIZE))
    non_normalized_count = len(feature_cols) - normalized_count
    
    print(f"\nNormalization breakdown:")
    print(f"  Features normalized (Z-score): {normalized_count}")
    print(f"  Features not normalized: {non_normalized_count}")
    print(f"  Normalized features: {FEATURES_TO_NORMALIZE}")
    print(f"  Non-normalized features: {NON_NORMALIZED_FEATURES}")
    
    # Calculate total feature count
    n_pipelines = len(PIPELINES)
    n_structures = len(STRUCTURE_LABELS)
    n_features_per_structure = len(FEATURES_TO_NORMALIZE) + len(NON_NORMALIZED_FEATURES)
    total_features = n_pipelines * n_structures * n_features_per_structure
    print(f"\nFeature breakdown:")
    print(f"  Pipelines: {n_pipelines}")
    print(f"  Structures: {n_structures}")
    print(f"  Features per structure: {n_features_per_structure}")
    print(f"  Total features per subject: {total_features}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute morphological features for subcortical segmentations")
    p.add_argument("--n_subjects", type=int, default=0,
                   help="Number of subjects to process (0 for all subjects)")
    p.add_argument("--parallel", action="store_true", help="Parallelize over subjects")
    p.add_argument("--max_threads", type=int, default=10,
                   help="Maximum number of subjects processed concurrently in parallel")
    args = p.parse_args()
    main(args.n_subjects, args.parallel, args.max_threads)

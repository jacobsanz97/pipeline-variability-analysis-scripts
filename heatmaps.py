#!/usr/bin/env python3
"""
Robust single-row / multi-slice pipeline heatmap generator.
"""

import os
from glob import glob
from collections import defaultdict
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")
PIPELINES = {
    "freesurfer741ants243": "aseg_MNI.nii.gz",
    "freesurfer8001ants243": "aseg_MNI.nii.gz",
    "fslanat6071ants243": "subcortical_seg_MNI_ANTs.nii.gz",
    "samseg8001ants243": "seg_MNI.nii.gz",
}
FSL_LABELS = [10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
TEST_SUBJECTS = None  # comment out or change to None to use all
OUTPUT_PNG = os.path.join(ROOT_DIR, "pipeline_heatmap_comparison_multislice.png")
OUTPUT_CSV = os.path.join(ROOT_DIR, "pipeline_consistency_statistics.csv")
SLICES_TO_SHOW = 3  # number of representative slices (axial)

# -----------------------------
# Helper functions
# -----------------------------
def find_files_for_pipeline(root, pipeline, target_name):
    pattern = os.path.join(root, f"**/{pipeline}/**/{target_name}")
    files = sorted(glob(pattern, recursive=True))
    if TEST_SUBJECTS:
        files = files[:TEST_SUBJECTS]
    return files

def extract_subject_id(file_path, pipeline):
    """
    Extract subject ID based on your directory structure:
    <dataset>/derivatives/<pipeline>/<version>/output/sub-XXXXX/...
    """
    parts = file_path.split(os.sep)
    try:
        out_idx = parts.index("output")
        subj = parts[out_idx + 1]  # first-level sub-XXXX folder under output
        return subj
    except ValueError:
        # fallback
        return None

def ensure_image_on_reference(img_path, ref_img):
    img = nib.load(img_path)
    same_shape = img.shape == ref_img.shape
    same_affine = np.allclose(img.affine, ref_img.affine, atol=1e-6)
    arr = img.get_fdata(dtype=np.float32).astype(int)
    if same_shape and same_affine:
        return arr
    # resample
    resampled = resample_from_to(img, (ref_img.shape, ref_img.affine))
    arr = resampled.get_fdata(dtype=np.float32).astype(int)
    return arr

# -----------------------------
# Gather files and unique IDs
# -----------------------------
pipeline_subj_map = {}
for pl, fname in PIPELINES.items():
    files = find_files_for_pipeline(ROOT_DIR, pl, fname)
    subj_map = {}
    for f in files:
        parts = f.split(os.sep)
        try:
            dataset_idx = parts.index(os.path.basename(ROOT_DIR)) + 1
            dataset_name = parts[dataset_idx]
            out_idx = parts.index("output")
            subj_id = parts[out_idx + 1]
            unique_id = f"{dataset_name}_{subj_id}"
            subj_map[unique_id] = f
        except ValueError:
            continue
    pipeline_subj_map[pl] = subj_map
    print(f"{pl}: found {len(files)} total files, {len(subj_map)} unique subjects")

# Intersect subjects across all pipelines
all_subjects_sets = [set(subj_map.keys()) for subj_map in pipeline_subj_map.values()]
matched_subjects = set.intersection(*all_subjects_sets)
print(f"\n Found {len(matched_subjects)} subjects with all 4 pipelines available")

if len(matched_subjects) == 0:
    raise RuntimeError("No subjects with all 4 pipelines found.")

# -----------------------------
# Filter files to only matched subjects
# -----------------------------
pipeline_files = {}
for pl in PIPELINES:
    pipeline_files[pl] = [pipeline_subj_map[pl][s] for s in matched_subjects]
    print(f"{pl}: using {len(pipeline_files[pl])} matched subjects")

# -----------------------------
# Pick reference image from first matched subject
# -----------------------------
ref_img = nib.load(pipeline_files[list(PIPELINES.keys())[0]][0])
print(f"\nReference chosen: {ref_img.shape}")

# -----------------------------
# Compute frequency maps and statistics
# -----------------------------
heatmaps = {}
statistics = []

for pl, files in pipeline_files.items():
    acc = np.zeros(ref_img.shape, dtype=np.float32)
    print(f"\nProcessing pipeline '{pl}' ({len(files)} subjects)...")
    
    for fpath in tqdm(files, desc=pl):
        arr = ensure_image_on_reference(fpath, ref_img)
        mask = np.isin(arr, FSL_LABELS)
        acc += mask.astype(np.float32)
    
    acc /= len(files)
    heatmaps[pl] = acc
    
    # Compute statistics for this pipeline
    non_zero_voxels = acc[acc > 0]
    stats = {
        'pipeline': pl,
        'n_subjects': len(files),
        'max_frequency': float(np.max(acc)),
        'mean_frequency_nonzero': float(np.mean(non_zero_voxels)) if len(non_zero_voxels) > 0 else 0,
        'median_frequency_nonzero': float(np.median(non_zero_voxels)) if len(non_zero_voxels) > 0 else 0,
        'std_frequency_nonzero': float(np.std(non_zero_voxels)) if len(non_zero_voxels) > 0 else 0,
        'n_voxels_100%': int(np.sum(acc == 1.0)),
        'n_voxels_75%': int(np.sum(acc >= 0.75)),
        'n_voxels_50%': int(np.sum(acc >= 0.5)),
        'n_voxels_25%': int(np.sum(acc >= 0.25)),
        'n_voxels_any': int(np.sum(acc > 0)),
        'total_voxels': int(np.prod(acc.shape))
    }
    statistics.append(stats)
    
    print(f"  Max frequency: {stats['max_frequency']:.3f}")
    print(f"  Voxels with 100% agreement: {stats['n_voxels_100%']}")
    print(f"  Voxels with any signal: {stats['n_voxels_any']}")

# Save statistics to CSV
df_stats = pd.DataFrame(statistics)
df_stats.to_csv(OUTPUT_CSV, index=False)
print(f"\n Saved pipeline statistics to: {OUTPUT_CSV}")

# Print summary table
print("\n" + "="*80)
print("PIPELINE CONSISTENCY SUMMARY")
print("="*80)
for stats in statistics:
    print(f"{stats['pipeline']:25} | Max: {stats['max_frequency']:5.3f} | "
          f"100%: {stats['n_voxels_100%']:6d} | 75%+: {stats['n_voxels_75%']:6d} | "
          f"50%+: {stats['n_voxels_50%']:6d} | Any: {stats['n_voxels_any']:6d}")

# -----------------------------
# Choose representative slices
# -----------------------------
best_slices = [59, 74, 87]  # Your hardcoded slices

# -----------------------------
# Plot multi-row figure with slice labels on the left
# -----------------------------
n_pipelines = len(heatmaps)
fig, axes = plt.subplots(SLICES_TO_SHOW, n_pipelines, figsize=(5*n_pipelines, 5*SLICES_TO_SHOW))
if SLICES_TO_SHOW == 1:
    axes = [axes]

# Compute global vmax for visibility
vmax = 1.0
print(f"\nUsing vmax={vmax:.3f} for color scaling")

for r, sl_idx in enumerate(best_slices[::-1]):  # highest slice first
    # Compute MNI z-coordinate
    mni_coords = nib.affines.apply_affine(ref_img.affine, [[0,0,sl_idx]])[0]
    slice_label = f"Slice {sl_idx}/{ref_img.shape[2]}  z={mni_coords[2]:.1f}mm"
    
    for c, (pl, hm) in enumerate(heatmaps.items()):
        ax = axes[r, c] if SLICES_TO_SHOW > 1 else axes[c]
        slice_img = np.rot90(hm[:, :, sl_idx])
        im = ax.imshow(slice_img, cmap="magma", vmin=0, vmax=vmax)
        
        # Only first column gets the slice label as a text inside the axes
        if c == 0:
            ax.text(-0.12, 0.5, slice_label, fontsize=9, rotation=90,
                    va='center', ha='center', transform=ax.transAxes)
        
        # Top title: pipeline name + n subjects
        if r == 0:
            ax.set_title(f"{pl}  n={len(pipeline_files[pl])}", fontsize=10)
        
        ax.axis("off")

# Shared colorbar on the right
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax, label="Frequency")
cbar.ax.set_ylabel("Frequency (proportion of subjects)", rotation=270, labelpad=15)

plt.subplots_adjust(right=0.9, hspace=0.3, wspace=0.05)
plt.suptitle("Subcortical Segmentation Consistency Within Each Pipeline", fontsize=14)
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\n Saved multi-slice heatmap to: {OUTPUT_PNG}")

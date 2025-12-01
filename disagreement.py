#!/usr/bin/env python3
"""
Pipeline disagreement heatmap generator - shows where pipelines disagree across subjects.
MODIFIED VERSION: Creates both 99th percentile and full-range visualizations, plus histograms
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
import matplotlib.gridspec as gridspec

# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")
MNI_TEMPLATE_PATH = os.path.join(ROOT_DIR, "MNI152_T1_1mm_Brain.nii.gz")  # Added MNI template
PIPELINES = {
    "freesurfer741ants243": "aseg_MNI.nii.gz",
    "freesurfer8001ants243": "aseg_MNI.nii.gz",
    "fslanat6071ants243": "subcortical_seg_MNI_ANTs.nii.gz",
    "samseg8001ants243": "seg_MNI.nii.gz",
}
FSL_LABELS = [10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
TEST_SUBJECTS = None  # comment out or change to None to use all
OUTPUT_PNG = os.path.join(ROOT_DIR, "pipeline_disagreement_heatmap_multislice.png")
OUTPUT_FULLRANGE_PNG = os.path.join(ROOT_DIR, "pipeline_disagreement_heatmap_fullrange.png")
OUTPUT_HISTOGRAM_PNG = os.path.join(ROOT_DIR, "pipeline_disagreement_histograms.png")
OUTPUT_DISAGREEMENT_CSV = os.path.join(ROOT_DIR, "pipeline_disagreement_statistics.csv")
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
matched_subjects = list(set.intersection(*all_subjects_sets))
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
# Load and prepare MNI template (MOVED AFTER ref_img IS DEFINED)
# -----------------------------
print("Loading MNI template...")
mni_template_img = nib.load(MNI_TEMPLATE_PATH)
# Resample MNI template to match our reference image space
mni_template_resampled = resample_from_to(mni_template_img, (ref_img.shape, ref_img.affine))
mni_template = mni_template_resampled.get_fdata()
print(f"MNI template shape: {mni_template.shape} (resampled to match reference)")

# -----------------------------
# Compute pipeline disagreement maps
# -----------------------------
print("\n Computing pipeline disagreements...")

# Define pipeline pairs to compare
pipeline_pairs = [
    ("freesurfer741ants243", "freesurfer8001ants243"),
    ("freesurfer741ants243", "fslanat6071ants243"),
    ("freesurfer741ants243", "samseg8001ants243"),
    ("freesurfer8001ants243", "fslanat6071ants243"),
    ("freesurfer8001ants243", "samseg8001ants243"),
    ("fslanat6071ants243", "samseg8001ants243")
]

disagreement_maps = {}
disagreement_stats = []
# Store max values for annotation
max_disagreement_values = {}

for pl1, pl2 in pipeline_pairs:
    disagreement_acc = np.zeros(ref_img.shape, dtype=np.float32)
    
    print(f"\nComputing {pl1} vs {pl2}...")
    
    for i in tqdm(range(len(matched_subjects)), desc=f"{pl1[:10]} vs {pl2[:10]}"):
        # Load both pipeline segmentations for this subject
        arr1 = ensure_image_on_reference(pipeline_files[pl1][i], ref_img)
        arr2 = ensure_image_on_reference(pipeline_files[pl2][i], ref_img)
        
        # Create binary masks for subcortical structures
        mask1 = np.isin(arr1, FSL_LABELS)
        mask2 = np.isin(arr2, FSL_LABELS)
        
        # Binary disagreement: where one pipeline says subcortical, the other doesn't
        disagreement = np.logical_xor(mask1, mask2)
        disagreement_acc += disagreement.astype(np.float32)
    
    # Convert to proportion of subjects with disagreement
    disagreement_acc /= len(matched_subjects)
    comparison_name = f"{pl1}_vs_{pl2}"
    disagreement_maps[comparison_name] = disagreement_acc
    
    # Store maximum for annotation
    max_disagreement_values[comparison_name] = float(np.max(disagreement_acc))
    
    # Compute statistics for this pair
    stats = {
        'comparison': comparison_name,
        'n_subjects': len(matched_subjects),
        'mean_disagreement': float(np.mean(disagreement_acc)),
        'max_disagreement': max_disagreement_values[comparison_name],
        'median_disagreement': float(np.median(disagreement_acc)),
        'std_disagreement': float(np.std(disagreement_acc)),
        'n_voxels_high_disagreement': int(np.sum(disagreement_acc >= 0.5)),  # ≥50% subjects disagree
        'n_voxels_medium_disagreement': int(np.sum(disagreement_acc >= 0.25)),  # ≥25% subjects disagree
        'n_voxels_any_disagreement': int(np.sum(disagreement_acc > 0)),
        'total_voxels': int(np.prod(disagreement_acc.shape))
    }
    disagreement_stats.append(stats)
    
    print(f"  Mean disagreement: {stats['mean_disagreement']:.3f}")
    print(f"  Max disagreement: {stats['max_disagreement']:.3f}")
    print(f"  Voxels with ≥50% disagreement: {stats['n_voxels_high_disagreement']}")

# Save disagreement statistics to CSV
df_disagreement = pd.DataFrame(disagreement_stats)
df_disagreement.to_csv(OUTPUT_DISAGREEMENT_CSV, index=False)
print(f"\n Saved pipeline disagreement statistics to: {OUTPUT_DISAGREEMENT_CSV}")

# Print summary table
print("\n" + "="*100)
print("PIPELINE DISAGREEMENT SUMMARY")
print("="*100)
for stats in disagreement_stats:
    print(f"{stats['comparison']:45} | Mean: {stats['mean_disagreement']:5.3f} | "
          f"Max: {stats['max_disagreement']:5.3f} | ≥50%: {stats['n_voxels_high_disagreement']:6d} | "
          f"≥25%: {stats['n_voxels_medium_disagreement']:6d} | Any: {stats['n_voxels_any_disagreement']:6d}")

# -----------------------------
# Choose representative slices
# -----------------------------
best_slices = [59, 74, 87]  # Your hardcoded slices

# -----------------------------
# FUNCTION: Create visualization with given vmax
# -----------------------------
def create_visualization(disagreement_maps, vmax_mode, filename_suffix=""):
    """Create visualization with specified vmax mode ('percentile99' or 'fullrange')"""
    
    n_comparisons = len(disagreement_maps)
    
    # Compute global vmax based on mode
    all_disagreements = np.concatenate([hm.ravel() for hm in disagreement_maps.values()])
    
    if vmax_mode == 'percentile99':
        vmax = np.percentile(all_disagreements, 99)
        vmax_label = f"99th percentile: {vmax:.3f}"
        title_suffix = " (99th Percentile Normalization)"
    else:  # fullrange
        vmax = np.max(all_disagreements)
        vmax_label = f"full range: {vmax:.3f}"
        title_suffix = " (Full Range)"
    
    print(f"\nCreating {vmax_mode} visualization: using vmax={vmax:.3f}")
    
    # Create figure
    fig, axes = plt.subplots(SLICES_TO_SHOW, n_comparisons, 
                            figsize=(4 * n_comparisons, 4 * SLICES_TO_SHOW))
    if SLICES_TO_SHOW == 1:
        axes = [axes]
    elif n_comparisons == 1:
        axes = [[ax] for ax in axes]
    
    for r, sl_idx in enumerate(best_slices[::-1]):  # highest slice first
        # Compute MNI z-coordinate
        mni_coords = nib.affines.apply_affine(ref_img.affine, [[0, 0, sl_idx]])[0]
        slice_label = f"Slice {sl_idx}/{ref_img.shape[2]}  z={mni_coords[2]:.1f}mm"
        
        for c, (comp_name, dm) in enumerate(disagreement_maps.items()):
            ax = axes[r][c] if SLICES_TO_SHOW > 1 else axes[c]
            
            # Plot MNI template as background
            template_slice = np.rot90(mni_template[:, :, sl_idx])
            ax.imshow(template_slice, cmap='gray', alpha=0.7)
            
            # Plot disagreement heatmap as overlay
            slice_img = np.rot90(dm[:, :, sl_idx])
            im = ax.imshow(slice_img, cmap="hot", vmin=0, vmax=vmax, alpha=0.8)
            
            # Only first column gets the slice label
            if c == 0:
                ax.text(-0.15, 0.5, slice_label, fontsize=9, rotation=90,
                        va='center', ha='center', transform=ax.transAxes)
            
            # Top title: comparison name
            if r == 0:
                # Shorten pipeline names for display
                pl1, pl2 = comp_name.split("_vs_")
                
                # Apply the new naming scheme
                pl1_short = pl1.replace('freesurfer741ants243', 'FS741')\
                               .replace('freesurfer8001ants243', 'FS8001')\
                               .replace('fslanat6071ants243', 'FSL6071')\
                               .replace('samseg8001ants243', 'Samseg8')
                pl2_short = pl2.replace('freesurfer741ants243', 'FS741')\
                               .replace('freesurfer8001ants243', 'FS8001')\
                               .replace('fslanat6071ants243', 'FSL6071')\
                               .replace('samseg8001ants243', 'Samseg8')
        
                ax.set_title(f"{pl1_short} vs {pl2_short}\nn={len(matched_subjects)}", fontsize=10, pad=10)
    
            ax.axis("off")
    
    # Shared colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label="Disagreement")
    
    # Add annotation about color scaling
    max_val = max_disagreement_values.get(comp_name, 0.989)  # Use actual or placeholder
    annotation_text = f"Color scale: {vmax_label}\nMax observed: {max_val:.3f}"
    fig.text(0.92, 0.05, annotation_text, fontsize=9, 
             ha='center', va='bottom', transform=fig.transFigure,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    cbar.ax.set_ylabel("Proportion of subjects with pipeline disagreement", 
                      rotation=270, labelpad=15)
    
    plt.subplots_adjust(right=0.9, hspace=0.2, wspace=0.1)
    plt.suptitle(f"Pipeline Disagreement Overlaid on MNI Template{title_suffix}", 
                 fontsize=14, y=0.95)
    
    # Save figure
    output_path = os.path.join(ROOT_DIR, f"pipeline_disagreement_heatmap_{vmax_mode}{filename_suffix}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved to: {output_path}")
    
    return output_path

# -----------------------------
# Create both visualizations
# -----------------------------
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# 1. 99th percentile visualization
output_99 = create_visualization(disagreement_maps, 'percentile99')

# 2. Full range visualization
output_full = create_visualization(disagreement_maps, 'fullrange')

# -----------------------------
# Create histogram figure
# -----------------------------
print("\nCreating histogram figure...")

# Create figure with subplots for each comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Get global statistics for overall histogram
all_disagreement_values = []
all_nonzero_disagreements = []

for idx, (comp_name, dm) in enumerate(disagreement_maps.items()):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    data = dm.ravel()
    
    # Filter out zeros for better visualization of distribution
    nonzero_data = data[data > 0]
    all_nonzero_disagreements.extend(nonzero_data)
    
    # Create histogram
    n, bins, patches = ax.hist(nonzero_data, bins=50, alpha=0.7, color='steelblue', 
                               edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for key percentiles
    ax.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='25% threshold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1, label='50% threshold')
    
    # Add statistics text
    stats_text = (f"Max: {np.max(data):.3f}\n"
                  f"Mean: {np.mean(data):.4f}\n"
                  f"Median: {np.median(data):.4f}\n"
                  f"Std: {np.std(data):.4f}\n"
                  f"Non-zero: {len(nonzero_data):,}\n"
                  f"≥50%: {np.sum(data >= 0.5):,}")
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    ax.set_xlabel('Disagreement Proportion')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_yscale('log')  # Log scale to see distribution better
    ax.set_title(f"{comp_name.split('_vs_')[0][:10]} vs {comp_name.split('_vs_')[1][:10]}", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Add overall histogram on last subplot if available
if len(disagreement_maps) < len(axes):
    ax = axes[len(disagreement_maps)]
    ax.hist(all_nonzero_disagreements, bins=100, alpha=0.7, color='darkgreen', 
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.set_xlabel('Disagreement Proportion')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_yscale('log')
    ax.set_title(f"All Comparisons Combined (n={len(all_nonzero_disagreements):,} non-zero voxels)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add overall stats
    overall_stats = (f"Global Max: {np.max(all_nonzero_disagreements):.3f}\n"
                     f"Global Mean: {np.mean(all_nonzero_disagreements):.4f}\n"
                     f"Global Median: {np.median(all_nonzero_disagreements):.4f}\n"
                     f"99th %ile: {np.percentile(all_nonzero_disagreements, 99):.3f}\n"
                     f"95th %ile: {np.percentile(all_nonzero_disagreements, 95):.3f}")
    
    ax.text(0.95, 0.95, overall_stats, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Distribution of Pipeline Disagreement Proportions (Non-Zero Voxels Only)", 
             fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_HISTOGRAM_PNG, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Saved histogram figure to: {OUTPUT_HISTOGRAM_PNG}")

# -----------------------------
# Print final summary
# -----------------------------
print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"1. 99th Percentile Visualization: {output_99}")
print(f"   - Shows patterns clearly without outlier domination")
print(f"   - vmax = 99th percentile of all disagreement values")
print(f"   - Recommended for seeing regional patterns")
print()
print(f"2. Full Range Visualization: {output_full}")
print(f"   - Shows complete range including extreme values")
print(f"   - vmax = maximum observed disagreement")
print(f"   - Reveals voxels with near-100% disagreement")
print()
print(f"3. Histogram Figure: {OUTPUT_HISTOGRAM_PNG}")
print(f"   - Shows distribution of disagreement values")
print(f"   - Log scale to visualize heavy-tailed distribution")
print(f"   - Includes key thresholds (25%, 50%)")
print()
print(f"4. Statistics CSV: {OUTPUT_DISAGREEMENT_CSV}")
print(f"   - Contains detailed statistics for each comparison")
print()

print(f"\n Key Insights from Maximum Values:")
for comp_name, max_val in max_disagreement_values.items():
    short_name = comp_name.replace('freesurfer741ants243', 'FS741')\
                          .replace('freesurfer8001ants243', 'FS8001')\
                          .replace('fslanat6071ants243', 'FSL6071')\
                          .replace('samseg8001ants243', 'Samseg8')
    print(f"  {short_name:40} max = {max_val:.3f} ({max_val*100:.1f}% of subjects)")

print(f"\n Interpretation:")
print("- Gray background = MNI anatomical template")
print("- Hot colors = regions where pipelines frequently disagree") 
print("- Cool colors = regions where pipelines usually agree")
print("- Histograms show bimodal distribution: most voxels agree perfectly, few disagree completely")
print("- Maximum disagreements near 1.0 indicate systematic algorithmic differences at specific locations")

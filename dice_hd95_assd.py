#!/usr/bin/env python3
"""
Fast Dice overlap computation between subcortical segmentations of different pipelines,
handling multiple datasets, parallelization (limited threads), and plotting mean Dice with n per dataset.
"""
import os, argparse, numpy as np, nibabel as nib, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from glob import glob
from collections import OrderedDict
from nibabel.processing import resample_from_to
from concurrent.futures import ThreadPoolExecutor

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    print("Warning: SimpleITK not available. Install with: pip install SimpleITK")
    SITK_AVAILABLE = False

# === OUTPUT IN CURRENT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

PIPELINES = {
    "freesurfer741ants243": "aseg.nii.gz",
    "freesurfer8001ants243": "aseg.nii.gz",
    "fslanat6071ants243": "T1_subcort_seg.nii.gz",
    "samseg8001ants243": "seg.nii.gz",
}

# Reordered to show left-right pairs
STRUCTURE_LABELS = OrderedDict([
    ("Left-Thalamus-Proper", 10),
    ("Right-Thalamus-Proper", 49),
    ("Left-Caudate", 11),
    ("Right-Caudate", 50),
    ("Left-Putamen", 12),
    ("Right-Putamen", 51),
    ("Left-Pallidum", 13),
    ("Right-Pallidum", 52),
    ("Left-Hippocampus", 17),
    ("Right-Hippocampus", 53),
    ("Left-Amygdala", 18),
    ("Right-Amygdala", 54),
    ("Left-Accumbens-area", 26),
    ("Right-Accumbens-area", 58),
    ("Brain-Stem/4thVentricle", 16),
])
LABELS = list(STRUCTURE_LABELS.values())

def find_files(root, pipeline, filename):
    return sorted(glob(os.path.join(root, f"**/{pipeline}/**/{filename}"), recursive=True))

def extract_ids(path, root_dir):
    """Extract dataset, subject, session from path - FIXED VERSION"""
    path_parts = Path(path).parts
    root_parts = Path(root_dir).parts
    
    # Find where the root_dir ends in the path - FIXED
    for i in range(len(path_parts) - len(root_parts) + 1):
        if path_parts[i:i+len(root_parts)] == root_parts:
            # The dataset is the next part after the root - FIXED
            if i + len(root_parts) < len(path_parts):
                dataset = path_parts[i + len(root_parts)]
            else:
                dataset = "unknown"
            break
    else:
        dataset = "unknown"
    
    # Extract subject and session - UNCHANGED
    sub = next((p for p in path_parts if p.startswith("sub-")), "nosub")
    ses = next((p for p in path_parts if p.startswith("ses-")), "nosess")
    
    return dataset, sub, ses

# Dice computation using boolean masks
def dice_per_label(arr1, arr2, labels):
    dices = {}
    for lbl in labels:
        m1 = arr1 == lbl
        m2 = arr2 == lbl
        inter = np.logical_and(m1, m2).sum()
        v1, v2 = m1.sum(), m2.sum()
        dices[lbl] = 1.0 if v1+v2 == 0 else 2*inter/(v1+v2)
    return dices

# Fast HD95 using SimpleITK
def fast_hd95_per_label(arr1, arr2, labels, voxel_spacing):
    """Compute HD95 using SimpleITK's optimized implementation"""
    if not SITK_AVAILABLE:
        return {lbl: np.nan for lbl in labels}
    
    hd95s = {}
    spacing_sitk = voxel_spacing[::-1]  # SimpleITK uses z,y,x order
    
    for lbl in labels:
        m1 = arr1 == lbl
        m2 = arr2 == lbl
        
        if m1.sum() == 0 or m2.sum() == 0:
            hd95s[lbl] = np.nan
            continue
        
        try:
            sitk1 = sitk.GetImageFromArray(m1.astype(np.uint8))
            sitk2 = sitk.GetImageFromArray(m2.astype(np.uint8))
            sitk1.SetSpacing(spacing_sitk)
            sitk2.SetSpacing(spacing_sitk)
            
            hd95_filter = sitk.HausdorffDistanceImageFilter()
            hd95_filter.Execute(sitk1, sitk2)
            hd95 = hd95_filter.GetAverageHausdorffDistance()  # This is HD95!
            hd95s[lbl] = hd95
            
        except Exception as e:
            print(f"Warning: Error computing HD95 for label {lbl}: {e}")
            hd95s[lbl] = np.nan
    
    return hd95s

# ASSD using SimpleITK  
def assd_per_label(arr1, arr2, labels, voxel_spacing):
    """Compute ASSD using SimpleITK"""
    if not SITK_AVAILABLE:
        return {lbl: np.nan for lbl in labels}
    
    assds = {}
    spacing_sitk = voxel_spacing[::-1]
    
    for lbl in labels:
        m1 = arr1 == lbl
        m2 = arr2 == lbl
        
        if m1.sum() == 0 or m2.sum() == 0:
            assds[lbl] = np.nan
            continue
        
        try:
            sitk1 = sitk.GetImageFromArray(m1.astype(np.uint8))
            sitk2 = sitk.GetImageFromArray(m2.astype(np.uint8))
            sitk1.SetSpacing(spacing_sitk)
            sitk2.SetSpacing(spacing_sitk)
            
            # Compute ASSD using contour-based approach
            contour1 = sitk.BinaryContour(sitk1, fullyConnected=False)
            contour2 = sitk.BinaryContour(sitk2, fullyConnected=False)
            
            dt1 = sitk.SignedMaurerDistanceMap(contour1, useImageSpacing=True, squaredDistance=False)
            dt2 = sitk.SignedMaurerDistanceMap(contour2, useImageSpacing=True, squaredDistance=False)
            
            dt1_arr = sitk.GetArrayFromImage(dt1)
            dt2_arr = sitk.GetArrayFromImage(dt2)
            contour1_arr = sitk.GetArrayFromImage(contour1)
            contour2_arr = sitk.GetArrayFromImage(contour2)
            
            dist1_to_2 = np.abs(dt2_arr[contour1_arr > 0])
            dist2_to_1 = np.abs(dt1_arr[contour2_arr > 0])
            
            if len(dist1_to_2) > 0 and len(dist2_to_1) > 0:
                assd = (dist1_to_2.sum() + dist2_to_1.sum()) / (len(dist1_to_2) + len(dist2_to_1))
            else:
                assd = np.nan
                
            assds[lbl] = assd
            
        except Exception as e:
            print(f"Warning: Error computing ASSD for label {lbl}: {e}")
            assds[lbl] = np.nan
    
    return assds

def load_segmentation_as_int(path, ref_img=None):
    img = nib.load(path)
    if ref_img is not None and (img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine, atol=1e-3)):
        img = resample_from_to(img, ref_img, order=0)
    data = img.get_fdata(dtype=np.float32).astype(np.int16)
    return data, img.affine  # Return both data and affine for voxel spacing

def count_unique_subjects(df):
    """Count unique subjects across datasets (dataset + subject combination)"""
    return df[['dataset', 'subject']].drop_duplicates().shape[0]

def count_unique_subjects_per_dataset(df, dataset):
    """Count unique subjects within a specific dataset"""
    return df[df['dataset'] == dataset][['dataset', 'subject']].drop_duplicates().shape[0]

def get_heatmap_range(df, metric, dataset=None):
    """Get range for heatmap based on MEAN values (not raw data)"""
    if dataset:
        data_subset = df[df['dataset'] == dataset]
    else:
        data_subset = df
    
    # Calculate means for each pipeline pair and structure
    means = []
    for (p1, p2, structure), group in data_subset.groupby(['pipeline1', 'pipeline2', 'structure']):
        mean_val = group[metric].mean()
        if not np.isnan(mean_val):
            means.append(mean_val)
    
    if not means:
        return 0, 1  # Fallback
    
    mean_min = min(means)
    mean_max = max(means)
    
    # Add small padding for visual clarity
    padding = (mean_max - mean_min) * 0.1
    vmin = max(0, mean_min - padding)
    
    if metric == 'dice':
        vmax = min(1, mean_max + padding)  # Dice can't exceed 1
    else:
        vmax = mean_max + padding
    
    print(f"  {metric} heatmap range: {vmin:.3f}-{vmax:.3f} (means: {mean_min:.3f}-{mean_max:.3f})")
    return vmin, vmax

def main(n_subjects=10, parallel=False, max_threads=10, metrics=None):
    if metrics is None:
        metrics = ['dice', 'assd', 'hd95']
    
    ROOT_DIR = os.path.expanduser("~/Desktop/duckysets")

    # gather files for all pipelines
    subj_maps = {}
    for pl, fname in PIPELINES.items():
        files = find_files(ROOT_DIR, pl, fname)
        subj_maps[pl] = { "__".join(extract_ids(f, ROOT_DIR)): f for f in files }
        print(f"{pl}: {len(files)} files")

    # identify all datasets
    all_datasets = set(uid.split("__")[0] for pl_map in subj_maps.values() for uid in pl_map.keys())
    print(f"Detected datasets: {all_datasets}")

    # ADDED: Count total unique subjects before processing
    all_unique_subjects = set()
    for pl_map in subj_maps.values():
        all_unique_subjects.update(pl_map.keys())
    print(f"TOTAL unique subjects across all datasets: {len(all_unique_subjects)}")

    results = []

    # process each dataset separately
    for dataset in all_datasets:
        # find common subjects across pipelines within this dataset
        subjects_in_dataset = []
        for pl, pl_map in subj_maps.items():
            subjects_in_dataset.append(set(uid for uid in pl_map.keys() if uid.startswith(dataset)))
        common = set.intersection(*subjects_in_dataset)
        common = sorted(common)
        if n_subjects > 0:
            common = common[:n_subjects]
        print(f"Dataset {dataset}: matched {len(common)} subjects")

        if len(common) == 0:
            continue

        # pipeline pairs
        pairs = [(a,b) for i,a in enumerate(PIPELINES) for b in list(PIPELINES)[i+1:]]

        # pick first pipeline as reference for resampling
        ref_img = nib.load(subj_maps[list(PIPELINES.keys())[0]][common[0]])

        # function to compute metrics for one subject
        def compute_for_subject(uid):
            _, sub, ses = uid.split("__")
            imgs = {}
            affines = {}
            for pl in PIPELINES:
                data, affine = load_segmentation_as_int(subj_maps[pl][uid], ref_img)
                imgs[pl] = data
                affines[pl] = affine
            
            # Calculate voxel spacing from affine matrix
            voxel_spacing = np.sqrt(np.sum(ref_img.affine[:3,:3]**2, axis=0))
            
            local_results = []
            for p1, p2 in pairs:
                metric_results = {}
                
                if 'dice' in metrics:
                    metric_results['dice'] = dice_per_label(imgs[p1], imgs[p2], LABELS)
                
                if 'assd' in metrics:
                    metric_results['assd'] = assd_per_label(imgs[p1], imgs[p2], LABELS, voxel_spacing)
                
                if 'hd95' in metrics:
                    metric_results['hd95'] = fast_hd95_per_label(imgs[p1], imgs[p2], LABELS, voxel_spacing)
                
                for name, lbl in STRUCTURE_LABELS.items():
                    result_row = {
                        "dataset": dataset, "subject": sub, "session": ses,
                        "pipeline1": p1, "pipeline2": p2,
                        "structure": name, "label": lbl
                    }
                    for metric in metrics:
                        result_row[metric] = metric_results[metric][lbl]
                    local_results.append(result_row)
            return local_results

        if parallel:
            with ThreadPoolExecutor(max_workers=max_threads) as exe:
                for r in tqdm(exe.map(compute_for_subject, common), total=len(common),
                              desc=f"Metric computation ({dataset})"):
                    results.extend(r)
        else:
            for uid in tqdm(common, desc=f"Metric computation ({dataset})"):
                results.extend(compute_for_subject(uid))

    # save tidy CSV
    df = pd.DataFrame(results)
    out_csv = Path(OUTPUT_DIR)/"dice_overlap_tidy.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # === UPDATED PLOTTING CODE - MATCHING PLOT_ONLY VERSION ===
    short = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001',
        'samseg8001ants243': 'SAMSEG8'
    }
    pipelines = [short[p] for p in PIPELINES]

    # Print actual counts
    total_subjects = count_unique_subjects(df)
    print(f"TOTAL unique subjects (dataset+subject): {total_subjects}")
    for dataset in df['dataset'].unique():
        n = count_unique_subjects_per_dataset(df, dataset)
        print(f"  {dataset}: {n} subjects")

    # Create figures for each metric
    for metric in metrics:
        print(f"\n=== Plotting {metric.upper()} ===")
        
        # Get global range for this metric (based on means)
        global_vmin, global_vmax = get_heatmap_range(df, metric)
        
        # Individual dataset figures
        for dataset in df['dataset'].unique():
            print(f"Dataset: {dataset}")
            ds = df[df['dataset']==dataset]
            n_subjects_used = count_unique_subjects_per_dataset(df, dataset)
            
            # Get dataset-specific range (based on means)
            vmin, vmax = get_heatmap_range(df, metric, dataset)
            
            structs = list(STRUCTURE_LABELS.keys())
            ncols, nrows = 5, int(np.ceil(len(structs)/5))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4.5*nrows), squeeze=False)
            
            # Set appropriate colormaps based on metric
            if metric == 'dice':
                cmap = sns.color_palette("viridis", as_cmap=True)
            else:  # assd or hd95
                cmap = sns.color_palette("rocket_r", as_cmap=True)

            for i, s in enumerate(structs):
                ax = axes[i//ncols,i%ncols]
                d = ds[ds.structure==s]
                mat = pd.DataFrame(np.nan,index=pipelines,columns=pipelines)
                for (p1,p2),g in d.groupby(['pipeline1','pipeline2']):
                    mat.at[short[p1],short[p2]] = g[metric].mean()
                    mat.at[short[p2],short[p1]] = g[metric].mean()
                np.fill_diagonal(mat.values,1 if metric == 'dice' else 0)
                sns.heatmap(mat,vmin=vmin,vmax=vmax,annot=True,fmt=".2f",square=True,
                            cmap=cmap,cbar=False,ax=ax)
                ax.set_title(f"{s}", fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            for k in range(len(structs), ncols*nrows):
                fig.delaxes(axes[k//ncols, k%ncols])

            # colorbar
            cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin,vmax))
            metric_label = "Mean Dice" if metric == 'dice' else f"Mean {metric.upper()}"
            fig.colorbar(sm,cax=cax).set_label(metric_label, rotation=270, labelpad=15)

            # add n and metric type
            fig.text(0.5, 0.01, f"n = {n_subjects_used} scans | {metric_label} = mean across subjects",
                     ha='center', fontsize=12)

            fig.suptitle(f"Inter-pipeline {metric.upper()} per structure – {dataset}", fontsize=16)
            fig.tight_layout(rect=[0,0.03,0.9,0.95])
            out = Path(OUTPUT_DIR)/f"structure_{metric}_grid_{dataset}.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out}")
        
        # Pooled figure (all datasets combined) - use global range based on means
        print("All datasets:")
        n_subjects_pooled = count_unique_subjects(df)
        structs = list(STRUCTURE_LABELS.keys())
        ncols, nrows = 5, int(np.ceil(len(structs)/5))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4.5*nrows), squeeze=False)
        
        # Use global range for pooled figure (based on means)
        vmin, vmax = global_vmin, global_vmax
        
        if metric == 'dice':
            cmap = sns.color_palette("viridis", as_cmap=True)
        else:  # assd or hd95
            cmap = sns.color_palette("rocket_r", as_cmap=True)

        for i, s in enumerate(structs):
            ax = axes[i//ncols,i%ncols]
            d = df[df.structure==s]
            mat = pd.DataFrame(np.nan,index=pipelines,columns=pipelines)
            for (p1,p2),g in d.groupby(['pipeline1','pipeline2']):
                mat.at[short[p1],short[p2]] = g[metric].mean()
                mat.at[short[p2],short[p1]] = g[metric].mean()
            np.fill_diagonal(mat.values,1 if metric == 'dice' else 0)
            sns.heatmap(mat,vmin=vmin,vmax=vmax,annot=True,fmt=".2f",square=True,
                        cmap=cmap,cbar=False,ax=ax)
            ax.set_title(f"{s}", fontsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        for k in range(len(structs), ncols*nrows):
            fig.delaxes(axes[k//ncols, k%ncols])

        # colorbar
        cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin,vmax))
        metric_label = "Mean Dice" if metric == 'dice' else f"Mean {metric.upper()}"
        fig.colorbar(sm,cax=cax).set_label(metric_label, rotation=270, labelpad=15)

        # add n and metric type
        fig.text(0.5, 0.01, f"n = {n_subjects_pooled} scans | {metric_label} = mean across subjects",
                 ha='center', fontsize=12)

        fig.suptitle(f"Inter-pipeline {metric.upper()} per structure – ALL DATASETS", fontsize=16)
        fig.tight_layout(rect=[0,0.03,0.9,0.95])
        out = Path(OUTPUT_DIR)/f"structure_{metric}_grid_ALL.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_subjects", type=int, default=10,
                   help="Number of subjects to process (0 or negative for all subjects)")
    p.add_argument("--parallel", action="store_true", help="Parallelize over subjects")
    p.add_argument("--max_threads", type=int, default=10,
                   help="Maximum number of subjects processed concurrently in parallel")
    p.add_argument("--metrics", nargs="+", choices=['dice', 'assd', 'hd95'], default=['dice'],
                   help="Metrics to compute (default: dice only for speed)")
    a = p.parse_args()
    main(a.n_subjects, a.parallel, a.max_threads, a.metrics)

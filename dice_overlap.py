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

# === OUTPUT IN CURRENT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

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
LABELS = list(STRUCTURE_LABELS.values())

def find_files(root, pipeline, filename):
    return sorted(glob(os.path.join(root, f"**/{pipeline}/**/{filename}"), recursive=True))

def extract_ids(path, root_dir):
    parts = Path(path).parts
    try:
        root_index = parts.index(Path(root_dir).name)
        dataset = parts[root_index + 1]
    except ValueError:
        dataset = "unknown"
    sub = next((p for p in parts if p.startswith("sub-")), "nosub")
    ses = next((p for p in parts if p.startswith("ses-")), "nosess")
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

def load_segmentation_as_int(path, ref_img=None):
    img = nib.load(path)
    if ref_img is not None and (img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine, atol=1e-3)):
        img = resample_from_to(img, ref_img, order=0)
    data = img.get_fdata(dtype=np.float32).astype(np.int16)
    return data

def main(n_subjects=10, parallel=False, max_threads=10):
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

        # function to compute Dice for one subject
        def compute_for_subject(uid):
            _, sub, ses = uid.split("__")
            imgs = {pl: load_segmentation_as_int(subj_maps[pl][uid], ref_img) for pl in PIPELINES}
            local_results = []
            for p1, p2 in pairs:
                dvals = dice_per_label(imgs[p1], imgs[p2], LABELS)
                for name, lbl in STRUCTURE_LABELS.items():
                    local_results.append({
                        "dataset": dataset, "subject": sub, "session": ses,
                        "pipeline1": p1, "pipeline2": p2,
                        "structure": name, "label": lbl, "dice": dvals[lbl]
                    })
            return local_results

        if parallel:
            with ThreadPoolExecutor(max_workers=max_threads) as exe:
                for r in tqdm(exe.map(compute_for_subject, common), total=len(common),
                              desc=f"Dice computation ({dataset})"):
                    results.extend(r)
        else:
            for uid in tqdm(common, desc=f"Dice computation ({dataset})"):
                results.extend(compute_for_subject(uid))

    # save tidy CSV
    df = pd.DataFrame(results)
    out_csv = Path(OUTPUT_DIR)/"dice_overlap_tidy.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # plotting per dataset
    short = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001',
        'samseg8001ants243': 'SAMSEG8'
    }
    pipelines = [short[p] for p in PIPELINES]

    for dataset in df['dataset'].unique():
        ds = df[df['dataset']==dataset]
        n_subjects_used = ds['subject'].nunique()
        structs = list(STRUCTURE_LABELS.keys())
        ncols, nrows = 5, int(np.ceil(len(structs)/5))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4.5*nrows), squeeze=False)
        vmin,vmax=0,1
        cmap=sns.color_palette("viridis", as_cmap=True)

        for i, s in enumerate(structs):
            ax = axes[i//ncols,i%ncols]
            d = ds[ds.structure==s]
            mat = pd.DataFrame(np.nan,index=pipelines,columns=pipelines)
            for (p1,p2),g in d.groupby(['pipeline1','pipeline2']):
                mat.at[short[p1],short[p2]] = g.dice.mean()
                mat.at[short[p2],short[p1]] = g.dice.mean()
            np.fill_diagonal(mat.values,1)
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
        fig.colorbar(sm,cax=cax).set_label("Mean Dice", rotation=270, labelpad=15)

        # add n and Dice type
        fig.text(0.5, 0.01, f"n = {n_subjects_used} scans | Dice = mean across subjects",
                 ha='center', fontsize=12)

        fig.suptitle(f"Inter-pipeline Dice per structure – {dataset}", fontsize=16)
        fig.tight_layout(rect=[0,0.03,0.9,0.95])
        out = Path(OUTPUT_DIR)/f"structure_dice_grid_{dataset}.png"
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
    a = p.parse_args()
    main(a.n_subjects, a.parallel, a.max_threads)


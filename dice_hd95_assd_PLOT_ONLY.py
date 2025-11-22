#!/usr/bin/env python3
"""
Plotting-only version of dice_hd95_assd.py - generates plots from existing CSV results
"""
import os, argparse, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# === OUTPUT IN CURRENT DIRECTORY ===
OUTPUT_DIR = os.getcwd()

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

def main(csv_file, metrics=None):
    if metrics is None:
        # Auto-detect available metrics from CSV
        df = pd.read_csv(csv_file)
        available_metrics = [col for col in ['dice', 'hd95', 'assd'] if col in df.columns]
        metrics = available_metrics if available_metrics else ['dice']
        print(f"Auto-detected metrics: {metrics}")
    
    df = pd.read_csv(csv_file)
    
    # Print actual counts
    total_subjects = count_unique_subjects(df)
    print(f"TOTAL unique subjects (dataset+subject): {total_subjects}")
    for dataset in df['dataset'].unique():
        n = count_unique_subjects_per_dataset(df, dataset)
        print(f"  {dataset}: {n} subjects")
    
    # Pipeline name mapping
    short = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001',
        'samseg8001ants243': 'SAMSEG8'
    }
    pipelines = [short[p] for p in short.keys() if p in set(df['pipeline1']).union(set(df['pipeline2']))]

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
    p.add_argument("csv_file", help="Input CSV file with computed metrics")
    p.add_argument("--metrics", nargs="+", choices=['dice', 'assd', 'hd95'], 
                   help="Metrics to plot (default: auto-detect from CSV)")
    a = p.parse_args()
    main(a.csv_file, a.metrics)

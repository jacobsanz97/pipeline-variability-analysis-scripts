import json
from pathlib import Path
import pandas as pd
import os # Import os for current working directory

# ----------------------------
# Config
# ----------------------------
# Set output directory to current working directory
EXPERIMENT_STATE_ROOT = Path.cwd() 
STATE_DIR = Path.cwd() # Set discovery root to current working directory

# Structures to exclude for SAMSEG
SAMSEG_EXCLUDE = {
    "Intra-Cranial",
    "Unknown",
    "Skull",
    "Soft-Nonbrain-Tissue", # Normalized name
    "CSF",
    "Fluid-Inside-Eyes",    # Normalized name
    "Right-vessel",
    "WM-hypointensities",
    "Left-vessel",
    "non-WM-hypointensities",
    "5th-Ventricle",
}

# --- ADDED: FreeSurfer structures to exclude ---
FREESURFER_EXCLUDE = {
    "WM-hypointensities",
    "Optic-Chiasm",
    "Right-vessel",
    "Left-vessel",
}
# -----------------------------------------------

# ----------------------------
# File parsers
# ----------------------------
def parse_fsl(path: Path):
    """Parses FSL JSON, keeping all volumes."""
    with open(path) as f:
        data = json.load(f)
    # Return all structures (all values are float)
    return {k: float(v) for k, v in data.items()}

def parse_samseg(path: Path):
    """Parses SAMSEG CSV, excluding specified structures, and normalizes names."""
    results = {}
    try:
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Normalize ROI names and filter
        for _, row in df.iterrows():
            roi = row['ROI']
            
            # Normalize ROI names for comparison (e.g., "Brain-Stem" or "Soft_Nonbrain_Tissue")
            roi_normalized = roi.replace("Brain-Stem", "Brainstem").replace("_", "-")
            
            if roi_normalized not in SAMSEG_EXCLUDE:
                # Use the normalized name in results
                results[roi_normalized] = float(row['volume_mm3'])
                
    except Exception as e:
        print(f"Error parsing SAMSEG CSV {path}: {e}")
        
    return results

def parse_freesurfer(path: Path):
    """Parses FreeSurfer stats, keeping only structures with volume >= 100, and normalizes names, and excludes specified structures."""
    results = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            struct_name = parts[4]
            volume = float(parts[3])
            
            # Normalize structure names to match others
            if struct_name == "Brain-Stem":
                struct_name = "Brainstem"
            
            # --- MODIFIED: Check for exclusion ---
            if struct_name in FREESURFER_EXCLUDE:
                continue
            # -------------------------------------
            
            # Apply volume threshold filter
            if volume >= 100.0:
                results[struct_name] = volume
    return results

PARSERS = {
    "fsl": parse_fsl,
    "samseg": parse_samseg,
    "freesurfer": parse_freesurfer,
}

# ----------------------------
# Discovery (No change)
# ----------------------------
# ----------------------------
# Discovery (MODIFIED)
# ----------------------------
def discover_files(state_dir: Path):
    """
    Recursively discovers files in the new directory structure,
    starting from state_dir (the current directory).
    """
    results = []
    # state_dir (CWD) contains dataset directories like ds002345_nipoppy
    for top_level_dir in state_dir.iterdir():
        # Check for the expected structure: top_level_dir / dataset_dir
        if not top_level_dir.is_dir() or top_level_dir.name.startswith('.'):
            continue
        
        # --- MODIFIED LOGIC: Look for any directory ending in '_nipoppy' ---
        dataset_dir = None
        dataset_name = None
        
        for inner_dir in top_level_dir.iterdir():
            # The inner directory name is the canonical dataset name (e.g., ds003592_nipoppy)
            if inner_dir.is_dir() and inner_dir.name.endswith("_nipoppy"):
                dataset_dir = inner_dir
                dataset_name = inner_dir.name # Use the canonical name for the column
                break
        
        if dataset_dir is None:
            continue
            
        derivatives = dataset_dir / "derivatives"
        if not derivatives.exists():
            continue
            
        # dataset_name is already set to inner_dir.name
        # dataset_name = dataset_dir.name # This line is now handled above
        
        for pipeline_root in derivatives.iterdir():
            if not pipeline_root.is_dir():
                continue
            pipeline_name = pipeline_root.name
            
            for version_dir in pipeline_root.iterdir():
                if not version_dir.is_dir():
                    continue
                version = version_dir.name
                output_dir = version_dir / "output"
                if not output_dir.exists():
                    continue
                    
                # output_dir contains subj_dir
                for subj_dir in output_dir.iterdir():
                    if not subj_dir.is_dir():
                        continue
                    subj = subj_dir.name
                    
                    # subj_dir contains ses_dir
                    for ses_dir in subj_dir.iterdir():
                        if not ses_dir.is_dir():
                            continue
                        ses = ses_dir.name
                        
                        # --- FreeSurfer ---
                        # Path structure: ses_dir / subj / stats / aseg.stats
                        fs_stats = ses_dir / subj / "stats" / "aseg.stats"
                        if fs_stats.exists():
                            results.append({
                                "dataset": dataset_name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "freesurfer",
                                "path": fs_stats,
                            })
                        
                        # --- SAMSEG ---
                        # Path structure: ses_dir / samseg / samseg.csv
                        samseg_csv = ses_dir / "samseg" / "samseg.csv"
                        if samseg_csv.exists():
                            results.append({
                                "dataset": dataset_name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "samseg",
                                "path": samseg_csv,
                            })
                        
                        # --- FSL ---
                        # Path structure: ses_dir / out.anat / subcortical_volumes.json
                        fsl_json = ses_dir / "out.anat" / "subcortical_volumes.json"
                        if fsl_json.exists():
                            results.append({
                                "dataset": dataset_name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "fsl",
                                "path": fsl_json,
                            })
    return results

# ----------------------------
# Build tidy DataFrame (No change)
# ----------------------------
def build_tidy_dataframe(files_meta):
    tidy_rows = []
    for r in files_meta:
        parser = PARSERS.get(r["file_type"])
        if parser is None:
            continue
        try:
            vols = parser(r["path"])
            for struct, vol in vols.items():
                tidy_rows.append({
                    **r,  # keep metadata
                    "structure": struct,
                    "volume_mm3": vol,
                })
        except Exception as e:
            print(f"Error parsing {r['path']}: {e}")
            continue
    return pd.DataFrame(tidy_rows)

# ----------------------------
# Wide pivot for ML (MODIFIED)
# ----------------------------
def pivot_wide(df_tidy: pd.DataFrame):
    df_wide = df_tidy.pivot_table(
        index=["dataset", "subject", "session"],
        columns=["pipeline", "structure"],
        values="volume_mm3"
    )
    
    # Drop rows where all elements are NaN (completely empty rows)
    df_wide = df_wide.dropna(axis=0, how='all')
    # Drop columns where all elements are NaN (completely empty columns)
    df_wide = df_wide.dropna(axis=1, how='all')
    
    # flatten MultiIndex columns
    df_wide.columns = [f"{pipe}__{struct}" for pipe, struct in df_wide.columns]
    df_wide = df_wide.reset_index()
    return df_wide

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(f"Output directory: {EXPERIMENT_STATE_ROOT.absolute()}")
    
    files_meta = discover_files(STATE_DIR)
    print(f"Discovered {len(files_meta)} stats files")

    df_tidy = build_tidy_dataframe(files_meta)
    print(f"Tidy DataFrame shape: {df_tidy.shape}")
    
    # Save tidy DataFrame to the experiment state directory (current dir)
    tidy_path = EXPERIMENT_STATE_ROOT / "df_tidy.csv"
    df_tidy.to_csv(tidy_path, index=False)
    print(f"Saved tidy DataFrame to: {tidy_path.absolute()}")

    df_wide = pivot_wide(df_tidy)
    print(f"Wide DataFrame shape: {df_wide.shape}")
    
    # Save wide DataFrame to the experiment state directory (current dir)
    wide_path = EXPERIMENT_STATE_ROOT / "morphological_features_aseg.csv"
    df_wide.to_csv(wide_path, index=False)
    print(f"Saved wide DataFrame to: {wide_path.absolute()}")
    
    # Print pipeline counts for verification
    print("\nPipeline counts:")
    print(df_tidy['pipeline'].value_counts())
    
    # Print file type counts for verification  
    print("\nFile type counts:")
    print(df_tidy['file_type'].value_counts())
    
    # Verify files were created
    print(f"\nVerifying file creation:")
    print(f"Tidy CSV exists: {tidy_path.exists()} ({tidy_path.stat().st_size} bytes)")
    print(f"Wide CSV exists: {wide_path.exists()} ({wide_path.stat().st_size} bytes)")

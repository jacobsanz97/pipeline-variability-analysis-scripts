import json
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# Config
# ----------------------------
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT
EXPERIMENT_STATE_ROOT = ROOT

# ----------------------------
# Age/Sex extraction logic - MODIFIED FOR STANDARDIZATION
# ----------------------------
def load_tabular_data(dataset_path: Path):
    """Load all TSVs in the tabular/ directory."""
    tabular = dataset_path / "tabular"
    if not tabular.exists():
        return {}

    dfs = {}
    for tsv in tabular.glob("*.tsv"):
        try:
            dfs[tsv.stem] = pd.read_csv(tsv, sep="\t")
        except Exception as e:
            print(f"Could not read {tsv}: {e}")
    return dfs

def extract_demographics(dataset_name: str, dfs: dict):
    """Dataset-specific logic to extract age and sex information - MODIFIED for standardization."""
    df_demo = None
    dataset_lower = dataset_name.lower()

    # PREVENT-AD logic
    if dataset_lower == 'preventad':
        if 'participants' in dfs and 'ad8' in dfs:
            part = dfs['participants']
            ad8 = dfs['ad8']
            ad8 = ad8.rename(columns={'participant_id': 'subject', 'Candidate_Age': 'age_months'})
            ad8['age'] = ad8['age_months'] / 12.0
            df_demo = ad8.merge(part, left_on='subject', right_on='participant_id', how='left')
            df_demo = df_demo[['subject', 'visit_id', 'age', 'sex']]
    elif dataset_lower == 'ds005752':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    elif dataset_lower == 'ds003592':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    else:
        if 'participants' in dfs:
            df_demo = dfs['participants'].rename(columns={'participant_id': 'subject'})
            if 'age' not in df_demo.columns:
                df_demo['age'] = np.nan
            if 'sex' not in df_demo.columns:
                df_demo['sex'] = np.nan
    
    # Handle comma-separated multiple values by taking the first one
    if df_demo is not None and not df_demo.empty:
        # For age: take first value and convert to float
        if 'age' in df_demo.columns:
            df_demo['age'] = df_demo['age'].astype(str).str.split(',').str[0]
            # Replace 'n/a' with NaN and convert to float
            df_demo['age'] = df_demo['age'].replace('n/a', np.nan)
            df_demo['age'] = pd.to_numeric(df_demo['age'], errors='coerce')
        
        # For sex: take first value
        if 'sex' in df_demo.columns:
            df_demo['sex'] = df_demo['sex'].astype(str).str.split(',').str[0]
            # Replace 'n/a' with NaN
            df_demo['sex'] = df_demo['sex'].replace('n/a', np.nan)

            # --- START Standardization logic (NEW) ---
            # 1. Convert non-NaN to string and uppercase for robust matching
            non_nan_mask = df_demo['sex'].notna()
            df_demo.loc[non_nan_mask, 'sex'] = (
                df_demo.loc[non_nan_mask, 'sex'].astype(str).str.upper()
            )
            
            # 2. Apply standardization map: convert MALE/FEMALE to M/F
            standardization_map = {
                'MALE': 'M',
                'FEMALE': 'F',
                # 'M' and 'F' are kept as is (handled by uppercase)
            }
            df_demo.loc[non_nan_mask, 'sex'] = (
                df_demo.loc[non_nan_mask, 'sex'].replace(standardization_map)
            )
            # --- END Standardization logic ---
    
    # Return subject, age, AND sex
    if df_demo is not None and not df_demo.empty:
        return df_demo[['subject', 'age', 'sex']]
    else:
        # Return all required columns as empty if no data found
        return pd.DataFrame(columns=['subject', 'age', 'sex'])

# ----------------------------
# Discover subjects and sessions - UNCHANGED
# ----------------------------
def discover_subjects(state_dir: Path):
    """Discover all subjects and sessions without parsing volume files."""
    results = []

    for dataset_outer in state_dir.iterdir():
        if not dataset_outer.is_dir():
            continue

        for dataset_dir in dataset_outer.iterdir():
            if not dataset_dir.is_dir():
                continue

            derivatives = dataset_dir / "derivatives"
            if not derivatives.exists():
                continue

            for pipeline_root in derivatives.iterdir():
                if not pipeline_root.is_dir():
                    continue

                for version_dir in pipeline_root.iterdir():
                    if not version_dir.is_dir():
                        continue

                    output_dir = version_dir / "output"
                    if not output_dir.exists():
                        continue

                    for subj_dir in output_dir.iterdir():
                        if not subj_dir.is_dir():
                            continue
                        subj = subj_dir.name

                        for ses_dir in subj_dir.iterdir():
                            if not ses_dir.is_dir():
                                continue
                            ses = ses_dir.name

                            # Just record the subject/session info
                            results.append({
                                "dataset": dataset_dir.name,
                                "subject": subj,
                                "session": ses,
                            })

    return results

# ----------------------------
# Main - UNCHANGED from previous modification
# ----------------------------
if __name__ == "__main__":
    print(f"Dataset root: {STATE_DIR}")
    print(f"Output CSV directory: {EXPERIMENT_STATE_ROOT}")

    # Discover all subjects and sessions
    subjects_meta = discover_subjects(STATE_DIR)
    print(f"Discovered {len(subjects_meta)} subject-session combinations")
    
    # Create initial DataFrame
    df_demo = pd.DataFrame(subjects_meta)
    
    # Add age AND sex column
    df_demo["age"] = np.nan
    df_demo["sex"] = np.nan 
    
    # Build age AND sex mapping
    age_mapping = {}
    sex_mapping = {} 
    
    for dataset in df_demo["dataset"].unique():
        dataset_path = None

        # Find the real dataset folder
        for outer in ROOT.iterdir():
            if outer.is_dir():
                inner = outer / dataset
                if inner.exists():
                    dataset_path = inner
                    break

        if dataset_path is None:
            continue

        dfs = load_tabular_data(dataset_path)
        # The extract_demographics function now returns 'subject', 'age', and 'sex'
        df_demo_dataset = extract_demographics(dataset, dfs)

        if df_demo_dataset is None or df_demo_dataset.empty:
            continue

        # Build mappings for this dataset
        for _, row in df_demo_dataset.iterrows():
            age_mapping[(dataset, row["subject"])] = row["age"]
            sex_mapping[(dataset, row["subject"])] = row["sex"] 

    # Apply the age AND sex mapping
    def get_age(row):
        return age_mapping.get((row["dataset"], row["subject"]), np.nan)
    
    def get_sex(row):
        return sex_mapping.get((row["dataset"], row["subject"]), np.nan)
    
    df_demo["age"] = df_demo.apply(get_age, axis=1)
    df_demo["sex"] = df_demo.apply(get_sex, axis=1) 
    
    # Remove duplicates
    df_demo = df_demo.drop_duplicates(subset=["dataset", "subject", "session"]).reset_index(drop=True)
    
    print(f"Demographics DataFrame shape: {df_demo.shape}")
    print(f"Subjects with age data: {df_demo['age'].notna().sum()}")
    print(f"Subjects with sex data: {df_demo['sex'].notna().sum()}")
    
    # Load morphological features and join
    features_path = ROOT / "morphological_features_mni.csv"
    if features_path.exists():
        print(f"Loading morphological features from: {features_path}")
        df_features = pd.read_csv(features_path)
        print(f"Morphological features shape: {df_features.shape}")
        
        # FIX THE TYPO: Change ds003592_nipoppyy to ds003592_nipoppy
        df_features['dataset'] = df_features['dataset'].replace('ds003592_nipoppyy', 'ds003592_nipoppy')
        
        # Join with demographics on dataset, subject, session
        df_merged = df_features.merge(
            df_demo[["dataset", "subject", "session", "age", "sex"]], 
            on=["dataset", "subject", "session"],
            how="left"
        )
        
        # Reorder columns: dataset, subject, session, age, sex, then all feature columns
        demo_columns = ["dataset", "subject", "session", "age", "sex"]
        feature_columns = [col for col in df_merged.columns if col not in demo_columns]
        df_merged = df_merged[demo_columns + feature_columns]
        
        print(f"Merged DataFrame shape: {df_merged.shape}")
        print(f"Subjects with age data in merged set: {df_merged['age'].notna().sum()}")
        print(f"Subjects with sex data in merged set: {df_merged['sex'].notna().sum()}")
        
        # Save the merged result
        merged_output_path = EXPERIMENT_STATE_ROOT / "ml_dataset_with_age_sex.csv"
        df_merged.to_csv(merged_output_path, index=False)
        print(f"Saved merged ML dataset to: {merged_output_path}")
        
        # Show sample
        print("\nSample of merged data:")
        print(df_merged[["dataset", "subject", "session", "age", "sex"]].head(10))
    else:
        print(f"Morphological features file not found at: {features_path}")
        # Save just demographics as fallback
        output_path = EXPERIMENT_STATE_ROOT / "demographics_with_sex.csv"
        df_demo.to_csv(output_path, index=False)
        print(f"Saved demographics DataFrame to: {output_path}")

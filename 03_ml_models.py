import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 🎯 REQUIREMENT 1: BASELINE MEAN PREDICTOR CLASS
# ----------------------------
class MeanPredictor:
    """A baseline model that always predicts the mean age of the training set."""
    def __init__(self):
        self.mean_age = None

    def fit(self, X_train, y_train):
        # Calculate the mean age from the training data (essential for CV folds)
        self.mean_age = y_train.mean()
        return self

    def predict(self, X_test):
        # Always predict that mean value
        if self.mean_age is None:
            raise ValueError("Model must be fitted before prediction.")
        # X_test can be numpy array or dataframe, we only need its length
        return np.full(len(X_test), self.mean_age)

# ----------------------------
# Configuration
# ----------------------------

# Define the pipelines and their feature sets - 'all_pipelines' ADDED BACK
PIPELINES = {
    'freesurfer8001ants243': 'freesurfer8001ants243',
    'freesurfer741ants243': 'freesurfer741ants243', 
    'samseg8001ants243': 'samseg8001ants243',
    'fslanat6071ants243': 'fslanat6071ants243',
    'all_pipelines': None  # ADDED BACK: Will use all features
}

# --- ALL MODEL TYPES TO BE EVALUATED ---
MODELS_TO_EVALUATE = ['baseline', 'elasticnet', 'kneighbors', 'histgradientboosting','extratrees', 'svm']

# --- TOGGLE FOR OUTLIER REMOVAL (NEW) ---
RUN_OUTLIER_REMOVAL = False 

# --- ORIGINAL/UNTOUCHED GRIDS (for non-tree models) ---
ELASTIC_NET_PARAMS = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

SVM_PARAMS = {
    'regressor__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__kernel': ['linear', 'rbf'],
    'regressor__gamma': ['scale', 'auto', 0.01, 0.1]
}

MLP_PARAMS = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'regressor__learning_rate_init': [0.001, 0.01],
    'regressor__early_stopping': [True]
}

# --- SIMPLIFIED GRIDS FOR TREE MODELS ---
RANDOM_FOREST_PARAMS = {
    'regressor__n_estimators': [100, 200],  
    'regressor__max_depth': [10, 20, None],  
    'regressor__min_samples_split': [2, 10], 
    'regressor__min_samples_leaf': [1, 5]    
}

EXTRA_TREES_PARAMS = {
    'regressor__n_estimators': [100, 200], 
    'regressor__max_depth': [10, 20, None], 
    'regressor__min_samples_split': [2, 5], 
    'regressor__min_samples_leaf': [1, 2]   
}

HIST_GBM_PARAMS = {
    'regressor__max_iter': [100, 200], 
    'regressor__learning_rate': [0.01, 0.1], 
    'regressor__max_leaf_nodes': [31, 63], 
    'regressor__max_depth': [5, None] 
}

K_NEIGHBORS_PARAMS = {
    'regressor__n_neighbors': [3, 5, 7, 9, 11],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2] 
}

# ----------------------------
# Data Preparation (Toggleable Outlier Removal)
# ----------------------------

def clean_age_value(age):
    """Convert age values to numeric, handling special cases like '89+'"""
    if pd.isna(age):
        return np.nan
    if isinstance(age, str):
        age = age.replace('+', '').strip()
        try:
            return float(age)
        except ValueError:
            return np.nan
    return float(age)

def get_feature_cols(df):
    """Utility to get all feature columns, matching logic in get_features_for_pipeline"""
    # ADD 'sex' TO METADATA COLS
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id', 'sex']
    return [col for col in df.columns if col not in metadata_cols]

# --- OUTLIER REMOVAL FUNCTION (Identical) ---
def remove_outliers_mad(df_in, threshold_percent=0.10, mad_threshold=6.0): 
    """
    Remove participants if more than 'threshold_percent' (10%) of their brain features 
    are extreme outliers based on the Median Absolute Deviation (MAD) method (6.0 MAD).
    """
    df = df_in.copy()
    all_feature_cols = get_feature_cols(df)
    
    if len(all_feature_cols) == 0:
        print("  WARNING: No feature columns found for outlier removal.")
        return df
        
    print(f"  Removing participants with > {threshold_percent * 100:.0f}% extreme outliers (MAD method, threshold={mad_threshold})...")
    initial_count = len(df)
    
    medians = df[all_feature_cols].median()
    mad = (df[all_feature_cols] - medians).abs().median()
    
    lower_bound = medians - mad_threshold * mad
    upper_bound = medians + mad_threshold * mad
    
    outlier_mask = (df[all_feature_cols] < lower_bound) | (df[all_feature_cols] > upper_bound)
    
    df['outlier_count'] = outlier_mask.sum(axis=1)
    
    max_allowed_outliers = int(len(all_feature_cols) * threshold_percent)
    
    df_clean = df[df['outlier_count'] <= max_allowed_outliers].drop(columns=['outlier_count']).copy()
    
    removed_count = initial_count - len(df_clean)
    print(f"  Removed {removed_count} samples ({(removed_count/initial_count)*100:.1f}%) based on MAD outlier threshold.")
    
    return df_clean


def load_and_preprocess_data(file_path, age_file_path=None):
    """Load and preprocess the data - MODIFIED TO ENSURE 'sex' IS PRESENT"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    if age_file_path:
        print("Loading age data from separate file...")
        age_df = pd.read_csv(age_file_path)
        merge_cols = ['dataset', 'subject', 'session']
        df = pd.merge(df, age_df, on=merge_cols, how='inner')
    
    if 'age' not in df.columns:
        raise KeyError("'age' column not found in the data. Please provide age data.")
        
    # --- REQUIREMENT 2: CHECK FOR 'sex' COLUMN ---
    if 'sex' not in df.columns:
        raise KeyError("REQUIRED: 'sex' column (M/F) not found in the input data. Please check your CSV.")
    
    # Standard data cleanup
    print("Cleaning age data...")
    df['age'] = df['age'].apply(clean_age_value)
    df['subject_id'] = df['dataset'] + '_' + df['subject']
    
    # Filter 'sex' to only M and F
    df['sex'] = df['sex'].astype(str).str.upper().str.strip()
    initial_count_sex = len(df)
    df_clean = df[df['sex'].isin(['M', 'F'])].copy()
    print(f"After removing non-M/F sex entries: {len(df_clean)} samples (removed {initial_count_sex - len(df_clean)})")
    
    # ... (Rest of data cleaning remains the same)
    
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['age']).copy()
    print(f"After removing missing/invalid age: {len(df_clean)} samples (removed {initial_count - len(df_clean)})")
    
    # ... (Feature handling uses updated get_feature_cols)
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id', 'sex']
    all_feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    
    initial_count_age_clean = len(df_clean)
    df_clean = df_clean.dropna(subset=all_feature_cols)
    print(f"After removing missing feature data: {len(df_clean)} samples (removed {initial_count_age_clean - len(df_clean)})")
    
    # --- STEP: REMOVE OUTLIERS (NOW TOGGLEABLE) ---
    if RUN_OUTLIER_REMOVAL:
        df_clean = remove_outliers_mad(df_clean)
    else:
        print("Outlier removal skipped based on configuration.")
    
    # Final numeric cleanup
    for col in all_feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    final_count = len(df_clean)
    df_clean = df_clean.dropna(subset=all_feature_cols)
    print(f"After final numeric cleanup: {len(df_clean)} samples (removed {final_count - len(df_clean)})")
    
    print(f"Final dataset: {len(df_clean)} samples, {len(df_clean['subject_id'].unique())} unique subjects")
    print(f"Age range: {df_clean['age'].min():.1f} - {df_clean['age'].max():.1f} years")
    print(f"Age distribution - Mean: {df_clean['age'].mean():.1f}, Std: {df_clean['age'].std():.1f}")
    print(f"Sex distribution - M: {len(df_clean[df_clean['sex']=='M'])}, F: {len(df_clean[df_clean['sex']=='F'])}")
    
    return df_clean

def get_features_for_pipeline(df, pipeline_name):
    """Get the appropriate features for a given pipeline - UPDATED for 'all_pipelines'"""
    if pipeline_name == 'all_pipelines':
        # Use all feature columns defined in get_feature_cols
        return get_feature_cols(df) 
    pipeline_features = [col for col in df.columns if col.startswith(pipeline_name + '__')]
    return pipeline_features

# ----------------------------
# Normalization Functions (Identical)
# ----------------------------

class DatasetPipelineScaler:
    # ... (DatasetPipelineScaler class is unchanged)
    def __init__(self):
        self.scalers_ = {}  
        self.feature_means_ = {}
        self.feature_stds_ = {}
    
    def fit(self, X, features, datasets, pipeline_name):
        unique_datasets = np.unique(datasets)
        
        for dataset in unique_datasets:
            dataset_mask = datasets == dataset
            X_dataset = X[dataset_mask]
            
            if len(X_dataset) > 1:
                scaler = StandardScaler()
                scaler.fit(X_dataset)
                self.scalers_[(dataset, pipeline_name)] = scaler
                self.feature_means_[(dataset, pipeline_name)] = scaler.mean_
                self.feature_stds_[(dataset, pipeline_name)] = scaler.scale_
            else:
                self.scalers_[(dataset, pipeline_name)] = None
        
        return self
    
    def transform(self, X, features, datasets, pipeline_name):
        X_normalized = np.zeros_like(X)
        unique_datasets = np.unique(datasets)
        
        for dataset in unique_datasets:
            dataset_mask = datasets == dataset
            X_dataset = X[dataset_mask]
            
            if (dataset, pipeline_name) in self.scalers_ and self.scalers_[(dataset, pipeline_name)] is not None:
                X_normalized[dataset_mask] = self.scalers_[(dataset, pipeline_name)].transform(X_dataset)
            else:
                X_normalized[dataset_mask] = X_dataset
        
        return X_normalized

def normalize_features(df, features, pipeline_name, datasets, fit_scaler=None):
    """
    Normalize features using z-score normalization per dataset
    """
    if len(features) == 0:
        return df, fit_scaler
    
    X = df[features].values
    
    if fit_scaler is None:
        scaler = DatasetPipelineScaler()
        scaler.fit(X, features, datasets, pipeline_name)
        X_normalized = scaler.transform(X, features, datasets, pipeline_name)
    else:
        X_normalized = fit_scaler.transform(X, features, datasets, pipeline_name)
        scaler = fit_scaler
    
    df_normalized = df.copy()
    df_normalized[features] = X_normalized
    
    return df_normalized, scaler

# ----------------------------
# Modeling Functions (Modified for Baseline and Model Type)
# ----------------------------

def create_model_pipeline(model_type):
    """Create a pipeline with the specified model, or the MeanPredictor for 'baseline'"""
    if model_type == 'baseline':
        model = MeanPredictor() # Use the custom class
        param_grid = {} # No hyperparameters to tune
        search_class = None # No search needed
    elif model_type == 'elasticnet':
        model = ElasticNet(random_state=42, max_iter=50000)
        param_grid = ELASTIC_NET_PARAMS
        search_class = RandomizedSearchCV
    elif model_type == 'randomforest':
        model = RandomForestRegressor(random_state=42)
        param_grid = RANDOM_FOREST_PARAMS
        search_class = RandomizedSearchCV
    elif model_type == 'extratrees':
        model = ExtraTreesRegressor(random_state=42)
        param_grid = EXTRA_TREES_PARAMS
        search_class = RandomizedSearchCV
    elif model_type == 'histgradientboosting':
        model = HistGradientBoostingRegressor(random_state=42, verbose=0)
        param_grid = HIST_GBM_PARAMS
        search_class = RandomizedSearchCV
    elif model_type == 'svm':
        model = SVR()
        param_grid = SVM_PARAMS
        search_class = RandomizedSearchCV
    elif model_type == 'mlp':
        model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
        param_grid = MLP_PARAMS
        search_class = RandomizedSearchCV
    elif model_type == 'kneighbors':
        model = KNeighborsRegressor(n_jobs=-1)
        param_grid = K_NEIGHBORS_PARAMS
        search_class = RandomizedSearchCV
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Baseline model does not use a Pipeline or StandardScaler in this setup
    if model_type != 'baseline':
        pipeline = Pipeline([('regressor', model)])
    else:
        pipeline = model # MeanPredictor is the model itself
    
    return pipeline, param_grid, search_class

def nested_cv_evaluation(df, model_type, pipeline_name):
    """
    Perform nested CV with proper normalization, sex-splitting, and baseline handling.
    Returns: a list of dictionaries, one for Male results and one for Female results.
    """
    print(f"  Evaluating {model_type} on {pipeline_name}...")
    
    features = get_features_for_pipeline(df, pipeline_name)
    
    if len(features) == 0 and model_type != 'baseline':
        print(f"    WARNING: No features found for pipeline '{pipeline_name}'")
        return []
    
    print(f"    Features: {len(features) if model_type != 'baseline' else 'N/A'}")
    
    # Full data required for GroupKFold
    X_full = df[features + ['dataset', 'sex']] if model_type != 'baseline' else df[['dataset', 'sex']]
    y_full = df['age']
    groups_full = df['subject_id']
    
    outer_cv = GroupKFold(n_splits=5)
    
    # Store results for Male and Female separately
    sex_results = {'M': {'scores': [], 'models': []}, 
                   'F': {'scores': [], 'models': []}}
    
    model_pipeline, param_grid, search_class = create_model_pipeline(model_type)
    
    # Determine search parameters (only for non-baseline models)
    if search_class == RandomizedSearchCV:
        # Calculate max iterations based on product of all grid lengths
        total_param_combos = np.prod([len(v) for v in param_grid.values()])
        n_iter = min(50, total_param_combos)
        search_kwargs = {'n_iter': n_iter, 'random_state': 42}
        if n_iter > 1:
            print(f"    Using RandomizedSearchCV with n_iter={n_iter}")
        elif n_iter == 1:
            print(f"    Using 1-step RandomizedSearchCV (effectively GridSearchCV for single combo)")
    else:
        search_kwargs = {}


    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups_full)):
        print(f"    Fold {fold + 1}:", end=" ")
        
        X_train_full, X_test_full = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_full, y_test_full = y_full.iloc[train_idx], y_full.iloc[test_idx]
        groups_train = groups_full.iloc[train_idx]
        
        # --- REQUIREMENT 2: SPLIT DATA BY SEX ---
        for sex in ['M', 'F']:
            
            # --- Training Split ---
            train_mask = X_train_full['sex'] == sex
            X_train_sex = X_train_full[train_mask].drop(columns=['sex'])
            y_train_sex = y_train_full[train_mask]
            groups_train_sex = groups_train[train_mask]
            datasets_train_sex = X_train_sex['dataset']
            
            # --- Testing Split ---
            test_mask = X_test_full['sex'] == sex
            X_test_sex = X_test_full[test_mask].drop(columns=['sex'])
            y_test_sex = y_test_full[test_mask]
            datasets_test_sex = X_test_sex['dataset']

            if len(y_train_sex) < 2 or len(y_test_sex) == 0:
                print(f"({sex} - Too few samples)", end=" ")
                continue
                
            # --- NORMALIZATION AND MODEL TRAINING ---
            try:
                if model_type != 'baseline':
                    # Normalize features for non-baseline models
                    X_train_features = X_train_sex.drop(columns=['dataset'])
                    X_test_features = X_test_sex.drop(columns=['dataset'])
                    
                    train_normalized, scaler = normalize_features(
                        X_train_features, features, pipeline_name, datasets_train_sex
                    )
                    test_normalized, _ = normalize_features(
                        X_test_features, features, pipeline_name, datasets_test_sex, fit_scaler=scaler
                    )
                    
                    X_train_norm = train_normalized[features].values
                    X_test_norm = test_normalized[features].values
                    
                    # Inner CV Search
                    inner_cv = GroupKFold(n_splits=3)
                    search = search_class(
                        model_pipeline, param_grid, cv=inner_cv, 
                        scoring='neg_mean_absolute_error', n_jobs=-1,
                        error_score='raise', **search_kwargs
                    )
                    
                    # FIX 2: Convert groups_train_sex to values to prevent 'unhashable type' error
                    search.fit(X_train_norm, y_train_sex, groups=groups_train_sex.values)
                    
                    best_model = search.best_estimator_
                    y_pred = best_model.predict(X_test_norm)
                    
                    # FIX 1: Calculate MAE immediately after prediction
                    mae = mean_absolute_error(y_test_sex, y_pred)
                    
                    # Log results
                    best_params_str = ", ".join([f"{k.split('__')[-1]}: {v}" for k, v in search.best_params_.items()])
                    print(f"({sex} MAE={mae:.3f}, Best params: {best_params_str}, N={len(y_train_sex)})", end=" ")

                else:
                    # --- REQUIREMENT 1: BASELINE MODEL ---
                    # Baseline model fits on un-normalized age data
                    baseline_model = MeanPredictor()
                    baseline_model.fit(None, y_train_sex) 
                    y_pred = baseline_model.predict(X_test_sex)
                    best_model = baseline_model # Store the fitted MeanPredictor
                    
                    # FIX 1: Calculate MAE immediately after prediction
                    mae = mean_absolute_error(y_test_sex, y_pred)
                    
                    print(f"({sex} MAE={mae:.3f}, Mean={best_model.mean_age:.1f}, N={len(y_train_sex)})", end=" ")
                
                sex_results[sex]['scores'].append(mae)
                sex_results[sex]['models'].append(best_model)
                
            except Exception as e:
                # print(f"({sex} FAILED: {str(e)[:20]}...)", end=" ")
                # We save the error for debugging but print a simple FAILED message
                print(f"({sex} FAILED: {str(e).split(':')[-1].strip()[:20]}...)", end=" ")
                continue
        print("") # Newline after fold completion

    # Format the results for the calling function
    final_results = []
    for sex, res in sex_results.items():
        valid_scores = [s for s in res['scores'] if not np.isnan(s)]
        if valid_scores:
            # We need to ensure y_train_sex is available to get the final N count if all folds fail, 
            # but since we only append results if successful, we rely on the last successful fold's N.
            # However, for consistency in the final report, we will use the total N for that sex.
            final_results.append({
                'sex': sex,
                'mean_mae': np.mean(valid_scores),
                'std_mae': np.std(valid_scores),
                'successful_folds': len(valid_scores),
                'n_train_samples_per_fold': len(y_train_sex) if 'y_train_sex' in locals() and len(y_train_sex) > 0 else 0,
            })
    return final_results


def evaluate_ensemble(df, all_pipeline_results):
    """
    REQUIREMENT 3 & FIX: Evaluate ensembles using the correct Model-Type/Pipeline Ensemble.
    Male and Female ensembles are separate.
    """
    print(f"\n{'='*60}")
    print(f"ENSEMBLE EVALUATION (Model-Type x Pipeline)")
    print(f"{'='*60}")
    
    # 1. Group results by sex and model type
    ensemble_results = {'M': {}, 'F': {}}
    for row in all_pipeline_results:
        sex = row['sex']
        model_type = row['model']
        pipeline = row['pipeline']
        
        # Only include results that aren't already ensembles, the all-pipelines model, or baseline
        if pipeline in PIPELINES and pipeline != 'all_pipelines' and model_type != 'baseline':
             
            if model_type not in ensemble_results[sex]:
                ensemble_results[sex][model_type] = []
            
            ensemble_results[sex][model_type].append(row)

    # 2. FIX 3: Initialize final_ensemble_report as a dictionary
    final_ensemble_report = {}

    # 3. Outer CV loop for ensemble evaluation
    groups = df['subject_id']
    outer_cv = GroupKFold(n_splits=5)
    
    # Data structure for CV:
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id', 'sex']
    all_feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X_full = df[all_feature_cols + ['dataset', 'sex']] 
    y_full = df['age']
    groups_full = groups.values
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups_full)):
        print(f"  Ensemble Fold {fold + 1}:")
        
        X_train_full, X_test_full = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train_full, y_test_full = y_full.iloc[train_idx], y_full.iloc[test_idx]
        
        # 4. Evaluate ensembles for each SEX and MODEL_TYPE
        for sex in ['M', 'F']:
            
            # --- Testing Split ---
            test_mask = X_test_full['sex'] == sex
            X_test_sex = X_test_full[test_mask].drop(columns=['sex'])
            y_test_sex = y_test_full[test_mask]
            datasets_test_sex = X_test_sex['dataset']
            
            if len(y_test_sex) == 0:
                continue

            for model_type, models_info in ensemble_results[sex].items():
                
                # Exclude ensembles with fewer than 2 successful pipelines
                # models_info contains the results of successful folds, we need the total pipelines used
                if len(models_info) < 2:
                    continue

                weighted_predictions = np.zeros(len(y_test_sex))
                total_effective_weight = 0.0
                
                # Simple averaging (weight=1.0)
                weight = 1.0 
                
                # Re-train and predict for each pipeline/model that belongs to this ensemble
                for model_info in models_info:
                    pipeline = model_info['pipeline']
                    
                    # Re-split training data for this specific model
                    train_mask = X_train_full['sex'] == sex
                    X_train_sex = X_train_full[train_mask].drop(columns=['sex'])
                    y_train_sex = y_train_full[train_mask]
                    datasets_train_sex = X_train_sex['dataset']
                    
                    # Filter features for this pipeline
                    features = get_features_for_pipeline(df, pipeline)

                    # --- TRAINING ---
                    try:
                        model_pipeline, _, _ = create_model_pipeline(model_type)
                        
                        # In a real environment, you'd load the best model or re-train with best params.
                        # Since we cannot pass the best model, we re-train with default/best params.
                        
                        X_train_features = X_train_sex.drop(columns=['dataset'])
                        
                        train_normalized, scaler = normalize_features(
                            X_train_features, features, pipeline, datasets_train_sex
                        )
                        X_pipeline_train = train_normalized[features].values
                        model_pipeline.fit(X_pipeline_train, y_train_sex)
                        
                        # --- TESTING ---
                        X_test_features = X_test_sex.drop(columns=['dataset'])
                        test_normalized, _ = normalize_features(
                            X_test_features, features, pipeline, datasets_test_sex, fit_scaler=scaler
                        )
                        X_pipeline_test = test_normalized[features].values
                        pred = model_pipeline.predict(X_pipeline_test)
                        
                        weighted_predictions += pred * weight
                        total_effective_weight += weight
                        
                    except Exception as e:
                        # print(f"      Failed to train/predict for {sex} {model_type} on {pipeline}: {e}")
                        continue
                
                # --- CALCULATE ENSEMBLE MAE ---
                if total_effective_weight > 0.001:
                    ensemble_pred = weighted_predictions / total_effective_weight
                    mae = mean_absolute_error(y_test_sex, ensemble_pred)
                    
                    # Store fold score for this ensemble
                    ensemble_name = f"ensemble_{model_type}"
                    if ensemble_name not in final_ensemble_report:
                        final_ensemble_report[ensemble_name] = {'M': {'scores': []}, 'F': {'scores': []}}
                        
                    final_ensemble_report[ensemble_name][sex]['scores'].append(mae)
                    print(f"    {sex} {model_type} Ensemble MAE = {mae:.3f} (N={len(models_info)} pipelines)")

    # 5. Aggregate results across folds
    aggregated_results = []
    for ensemble_type, sex_data in final_ensemble_report.items():
        for sex in ['M', 'F']:
            valid_scores = [s for s in sex_data[sex]['scores'] if not np.isnan(s)]
            if valid_scores:
                aggregated_results.append({
                    'pipeline': 'ensemble',
                    'model': ensemble_type,
                    'sex': sex,
                    'mean_mae': np.mean(valid_scores),
                    'std_mae': np.std(valid_scores),
                    'n_features': 'multiple',
                    'n_samples': len(df[df['sex']==sex]),
                    'n_subjects': len(df[df['sex']==sex]['subject_id'].unique()),
                    'successful_folds': len(valid_scores)
                })
    
    return aggregated_results

# ----------------------------
# Main Execution 
# ----------------------------

def main():
    print("=" * 60)
    print("BRAIN AGE PREDICTION ANALYSIS - SEX SPLIT AND CORRECT ENSEMBLE")
    print("=" * 60)
    
    # Load data - UPDATE THIS PATH to your actual file with age data
    # Assuming 'ml_dataset_with_age.csv' is the file you originally uploaded/used
    df = load_and_preprocess_data('ml_dataset_with_age_sex.csv')
    
    if len(df) == 0:
        print("ERROR: No valid data remaining after preprocessing!")
        return
    
    all_results = []
    
    # --- STEP 1 & 2: Evaluate individual models with Baseline and Sex Split ---
    for pipeline_name in PIPELINES.keys():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {pipeline_name.upper()}")
        print(f"{'='*50}")
        
        # Features are calculated inside nested_cv_evaluation now
        
        for model_type in MODELS_TO_EVALUATE:
            # Returns a list of dicts for M/F results
            sex_results = nested_cv_evaluation(df, model_type, pipeline_name)
            
            for res in sex_results:
                if res['successful_folds'] > 0:
                    
                    all_results.append({
                        'pipeline': pipeline_name,
                        'model': model_type,
                        'sex': res['sex'],
                        'mean_mae': res['mean_mae'],
                        'std_mae': res['std_mae'],
                        'n_features': len(get_features_for_pipeline(df, pipeline_name)) if model_type != 'baseline' else 0,
                        'n_samples': len(df[df['sex']==res['sex']]),
                        'n_subjects': len(df[df['sex']==res['sex']]['subject_id'].unique()),
                        'successful_folds': res['successful_folds'],
                        'n_train_samples_per_fold': res['n_train_samples_per_fold'],
                        'best_params': 'N/A' # Placeholder: need modification to nested_cv_evaluation to return best_params
                    })
                    print(f"  {res['sex']} {model_type.upper():20} - Mean MAE: {res['mean_mae']:.3f} ± {res['std_mae']:.3f} (N={res['n_train_samples_per_fold']})")
                else:
                    print(f"  {res['sex']} {model_type.upper():20} - No valid results")
    
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        
        # --- STEP 3: Evaluate Corrected Ensembles ---
        # Pass the individual model results to the ensemble function
        ensemble_results = evaluate_ensemble(df, all_results)
        all_results.extend(ensemble_results)
        
        # Update results_df with ensemble results
        results_df = pd.DataFrame(all_results)
        
        # Print final report
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY (Split by Sex)")
        print("=" * 80)
        
        # Group and report
        report_pipelines = list(PIPELINES.keys()) + ['ensemble']
        
        for sex in ['M', 'F']:
            print(f"\n--- RESULTS FOR {sex.upper()} (N_total={len(df[df['sex']==sex])}) ---")
            
            sex_df = results_df[results_df['sex'] == sex].copy()
            if len(sex_df) == 0:
                print("No successful models for this sex.")
                continue

            for pipeline in report_pipelines:
                pipeline_results = sex_df[sex_df['pipeline'] == pipeline]
                if len(pipeline_results) > 0:
                    print(f"\n{pipeline.upper()}:\n")
                    pipeline_results = pipeline_results.sort_values(by='mean_mae')
                    for _, row in pipeline_results.iterrows():
                        # ADDED: N_train_samples_per_fold to report
                        n_train_display = f" (N_train: {row['n_train_samples_per_fold']})" if row['pipeline'] != 'ensemble' else ""
                        print(f"  {row['model']:25} MAE: {row['mean_mae']:.3f} ± {row['std_mae']:.3f} (folds: {row['successful_folds']}){n_train_display}")
        
        print("\n" + "=" * 80)
        print("BEST PERFORMING MODELS OVERALL")
        print("=" * 80)
        
        successful_results_df = results_df[results_df['successful_folds'] > 0]
        
        if len(successful_results_df) > 0:
            best_overall = successful_results_df.loc[successful_results_df['mean_mae'].idxmin()]
            print(f"Best Overall: {best_overall['sex']} - {best_overall['pipeline']} with {best_overall['model']}")
            print(f"MAE: {best_overall['mean_mae']:.3f} ± {best_overall['std_mae']:.3f}")
        
        # Save results
        results_df.to_csv('brain_age_results_sex_split_and_correct_ensembles.csv', index=False)
        print(f"\nDetailed results saved to: brain_age_results_sex_split_and_correct_ensembles.csv")
    else:
        print("No successful model evaluations!")
    
    return results_df

if __name__ == "__main__":
    results = main()

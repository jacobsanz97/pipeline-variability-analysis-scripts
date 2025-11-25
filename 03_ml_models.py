import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor # ADDED K-Neighbors
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configuration
# ----------------------------

# Define the pipelines and their feature sets - UPDATED for extended features
PIPELINES = {
    'freesurfer8001ants243': 'freesurfer8001ants243',
    'freesurfer741ants243': 'freesurfer741ants243', 
    'samseg8001ants243': 'samseg8001ants243',
    'fslanat6071ants243': 'fslanat6071ants243',
    'all_pipelines': None  # Will use all features
}

# --- TOGGLE FOR OUTLIER REMOVAL (NEW) ---
RUN_OUTLIER_REMOVAL = False # Set to True to re-enable the MAD outlier filtering

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
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id']
    return [col for col in df.columns if col not in metadata_cols]

# --- OUTLIER REMOVAL FUNCTION (Identical to V6) ---
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
    """Load and preprocess the data"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    if age_file_path:
        print("Loading age data from separate file...")
        age_df = pd.read_csv(age_file_path)
        merge_cols = ['dataset', 'subject', 'session']
        df = pd.merge(df, age_df, on=merge_cols, how='inner')
    
    if 'age' not in df.columns:
        raise KeyError("'age' column not found in the data. Please provide age data.")
    
    print("Cleaning age data...")
    df['age'] = df['age'].apply(clean_age_value)
    df['subject_id'] = df['dataset'] + '_' + df['subject']
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['age']).copy()
    print(f"After removing missing/invalid age: {len(df_clean)} samples (removed {initial_count - len(df_clean)})")
    
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id']
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
    
    return df_clean

def get_features_for_pipeline(df, pipeline_name):
    """Get the appropriate features for a given pipeline"""
    if pipeline_name == 'all_pipelines':
        metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id']
        return [col for col in df.columns if col not in metadata_cols]
    else:
        pipeline_features = [col for col in df.columns if col.startswith(pipeline_name + '__')]
        return pipeline_features

# ----------------------------
# Normalization Functions (Identical to original)
# ----------------------------

class DatasetPipelineScaler:
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
# Modeling Functions (Identical to original)
# ----------------------------

def create_model_pipeline(model_type):
    """Create a pipeline with the specified model"""
    if model_type == 'elasticnet':
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
    
    pipeline = Pipeline([
        ('regressor', model)
    ])
    
    return pipeline, param_grid, search_class

def nested_cv_evaluation(df, model_type, pipeline_name):
    """Perform nested CV with proper normalization to avoid data leakage"""
    print(f"  Evaluating {model_type} on {pipeline_name}...")
    
    features = get_features_for_pipeline(df, pipeline_name)
    
    if len(features) == 0:
        print(f"    WARNING: No features found for pipeline '{pipeline_name}'")
        return [np.nan], []
    
    print(f"    Features: {len(features)}")
    
    model_pipeline, param_grid, search_class = create_model_pipeline(model_type)
    
    X = df[features]
    y = df['age']
    groups = df['subject_id']
    datasets = df['dataset']
    
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    groups_array = groups.values if hasattr(groups, 'values') else groups
    datasets_array = datasets.values if hasattr(datasets, 'values') else datasets
    
    outer_cv = GroupKFold(n_splits=5)
    outer_scores = []
    best_models = []
    
    if search_class == RandomizedSearchCV:
        n_iter = min(50, np.prod([len(v) for v in param_grid.values()]))
        search_kwargs = {'n_iter': n_iter, 'random_state': 42}
        print(f"    Using RandomizedSearchCV with n_iter={n_iter}")
    else:
        search_kwargs = {}
        try:
            total_combinations = np.prod([len(v) for v in param_grid.values()])
        except Exception:
            total_combinations = '?'
        print(f"    Using GridSearchCV (testing all {total_combinations} combinations)")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_array, y_array, groups_array)):
        print(f"    Fold {fold + 1}:", end=" ")
        
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        groups_train = groups_array[train_idx]
        datasets_train = datasets_array[train_idx]
        datasets_test = datasets_array[test_idx]
        
        train_df = pd.DataFrame(X_train, columns=features)
        train_df['dataset'] = datasets_train
        test_df = pd.DataFrame(X_test, columns=features)
        test_df['dataset'] = datasets_test
        
        train_normalized, scaler = normalize_features(train_df, features, pipeline_name, datasets_train)
        test_normalized, _ = normalize_features(test_df, features, pipeline_name, datasets_test, fit_scaler=scaler)
        
        X_train_norm = train_normalized[features].values
        X_test_norm = test_normalized[features].values
        
        inner_cv = GroupKFold(n_splits=3)
        
        try:
            search = search_class(
                model_pipeline, param_grid, cv=inner_cv, 
                scoring='neg_mean_absolute_error', n_jobs=-1,
                error_score='raise', **search_kwargs
            )
            search.fit(X_train_norm, y_train, groups=groups_train)
            
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test_norm)
            mae = mean_absolute_error(y_test, y_pred)
            
            outer_scores.append(mae)
            best_models.append(best_model)
            
            print(f"MAE = {mae:.3f}, Best params: {search.best_params_}")
            
        except Exception as e:
            print(f"FAILED - {str(e)}")
            continue
    
    if len(outer_scores) >= 3:
        return outer_scores, best_models
    else:
        print(f"  WARNING: Only {len(outer_scores)} successful folds for {model_type}")
        return [np.nan], []

def evaluate_ensemble(results_df, df, ensemble_type='simple_average'):
    """
    Evaluate ensemble model using best individual models from each pipeline.
    Includes simple average and weighted average based on 1/MAE.
    """
    if ensemble_type == 'simple_average':
        print(f"\n{'='*50}")
        print(f"ENSEMBLE EVALUATION - SIMPLE AVERAGE")
        print(f"{'='*50}")
        
    elif ensemble_type == 'weighted_average':
        print(f"\n{'='*50}")
        print(f"ENSEMBLE EVALUATION - WEIGHTED AVERAGE (1/MAE)")
        print(f"{'='*50}")
    
    best_models_info = {}
    for pipeline in PIPELINES.keys():
        if pipeline == 'all_pipelines':
            continue
            
        # EXCLUDE FSL PIPELINE FOR WEIGHTED ENSEMBLE
        if ensemble_type == 'weighted_average' and pipeline == 'fslanat6071ants243':
            continue
            
        pipeline_results = results_df[results_df['pipeline'] == pipeline]
        if len(pipeline_results) > 0:
            best_idx = pipeline_results['mean_mae'].idxmin()
            best_model_info = pipeline_results.loc[best_idx]
            if best_model_info['successful_folds'] > 0 and best_model_info['mean_mae'] > 0:
                best_models_info[pipeline] = best_model_info
    
    if len(best_models_info) < 2:
        print(f"Not enough successful pipelines for {ensemble_type.replace('_', ' ')} ensemble (need at least 2 successful individual models)")
        return [np.nan], []
    
    weights = {}
    if ensemble_type == 'weighted_average':
        inverse_maes = {p: 1.0 / info['mean_mae'] for p, info in best_models_info.items()}
        sum_inverse_maes = sum(inverse_maes.values())
        weights = {p: inv_mae / sum_inverse_maes for p, inv_mae in inverse_maes.items()}
        print(f"Creating weighted average ensemble from {len(best_models_info)} best models.")
    else:
        weights = {p: 1.0 for p in best_models_info.keys()}
        print(f"Creating simple average ensemble from {len(best_models_info)} best models.")

    for pipeline, info in best_models_info.items():
        weight_info = f" (Weight: {weights[pipeline]:.3f})" if ensemble_type == 'weighted_average' else ""
        print(f"  {pipeline}: {info['model']} (MAE: {info['mean_mae']:.3f}){weight_info}")
    
    groups = df['subject_id']
    datasets = df['dataset']
    outer_cv = GroupKFold(n_splits=5)
    
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id']
    all_feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X_full = df[all_feature_cols] 
    y_array = df['age'].values
    groups_array = groups.values
    datasets_array = datasets.values
    
    outer_scores = []
    ensemble_models = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_array, groups_array)):
        print(f"  Ensemble Fold {fold + 1}:")
        
        X_train_full, X_test_full = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        datasets_train = datasets_array[train_idx]
        datasets_test = datasets_array[test_idx]
        
        individual_models_for_fold = []
        
        for pipeline, model_info in best_models_info.items():
            model_type = model_info['model']
            features = get_features_for_pipeline(df, pipeline)
            
            if len(features) == 0:
                continue
                
            X_train_pipeline = X_train_full.loc[:, features]
            train_df_pipeline = X_train_pipeline.copy()
            train_df_pipeline['dataset'] = datasets_train
            
            train_normalized, scaler = normalize_features(train_df_pipeline, features, pipeline, datasets_train)
            X_pipeline_train = train_normalized[features].values
            
            model_pipeline, _, _ = create_model_pipeline(model_type)
            
            try:
                model_pipeline.fit(X_pipeline_train, y_train)
                individual_models_for_fold.append({
                    'model': model_pipeline, 
                    'pipeline': pipeline, 
                    'features': features, 
                    'scaler': scaler,
                    'weight': weights[pipeline]
                })
                print(f"    Trained {model_type} for {pipeline}")
            except Exception as e:
                print(f"    Failed to train {model_type} for {pipeline}: {e}")
                continue
        
        if len(individual_models_for_fold) < 2:
            print(f"    Not enough models trained for ensemble in fold {fold + 1}")
            continue
        
        weighted_predictions = np.zeros(len(y_test))
        total_effective_weight = 0.0
        
        for item in individual_models_for_fold:
            model, pipeline, features, scaler, weight = item['model'], item['pipeline'], item['features'], item['scaler'], item['weight']
            
            X_test_pipeline = X_test_full.loc[:, features]
            test_df_pipeline = X_test_pipeline.copy()
            test_df_pipeline['dataset'] = datasets_test
            
            test_normalized, _ = normalize_features(test_df_pipeline, features, pipeline, datasets_test, fit_scaler=scaler)
            X_pipeline_test = test_normalized[features].values
            
            pred = model.predict(X_pipeline_test)
            weighted_predictions += pred * weight
            total_effective_weight += weight
        
        if total_effective_weight > 0.001:
            ensemble_pred = weighted_predictions / total_effective_weight
        else:
            print(f"    Fold {fold + 1}: Effective weight too low, skipping.")
            continue
            
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        outer_scores.append(mae)
        ensemble_models.append(individual_models_for_fold)
        
        print(f"    Fold {fold + 1}: Ensemble MAE = {mae:.3f}")
    
    if len(outer_scores) >= 3:
        return outer_scores, ensemble_models
    else:
        print(f"  WARNING: Only {len(outer_scores)} successful folds for {ensemble_type.replace('_', ' ')}")
        return [np.nan], []

# ----------------------------
# Main Execution 
# ----------------------------

def main():
    print("=" * 60)
    print("BRAIN AGE PREDICTION ANALYSIS - EXTENDED FEATURES")
    print("=" * 60)
    
    # Load data - UPDATE THIS PATH to your actual file with age data
    df = load_and_preprocess_data('ml_dataset_with_age.csv')
    
    if len(df) == 0:
        print("ERROR: No valid data remaining after preprocessing!")
        return
    
    results = []
    
    models_to_evaluate = ['elasticnet', 'kneighbors', 'histgradientboosting', 'extratrees', 'svm']
    
    for pipeline_name in PIPELINES.keys():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {pipeline_name.upper()}")
        print(f"{'='*50}")
        
        features = get_features_for_pipeline(df, pipeline_name)
        
        if len(features) == 0:
            print(f"  WARNING: No features found for pipeline '{pipeline_name}'")
            continue
            
        print(f"Features: {len(features)}, Samples: {len(df)}, Subjects: {len(df['subject_id'].unique())}")
        
        for model_type in models_to_evaluate:
            scores, models = nested_cv_evaluation(df, model_type, pipeline_name)
            
            if scores and not all(np.isnan(scores)):
                valid_scores = [s for s in scores if not np.isnan(s)]
                if valid_scores:
                    mean_mae = np.mean(valid_scores)
                    std_mae = np.std(valid_scores)
                    
                    results.append({
                        'pipeline': pipeline_name,
                        'model': model_type,
                        'mean_mae': mean_mae,
                        'std_mae': std_mae,
                        'n_features': len(features),
                        'n_samples': len(df),
                        'n_subjects': len(df['subject_id'].unique()),
                        'successful_folds': len(valid_scores)
                    })
                    
                    print(f"  {model_type.upper():20} - Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
                else:
                    print(f"  {model_type.upper():20} - No valid results")
            else:
                print(f"  {model_type.upper():20} - Evaluation failed")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        
        # 1. Simple Average Ensemble
        ensemble_scores, _ = evaluate_ensemble(results_df, df, ensemble_type='simple_average')
        
        if ensemble_scores and not all(np.isnan(ensemble_scores)):
            valid_scores = [s for s in ensemble_scores if not np.isnan(s)]
            if valid_scores:
                mean_mae = np.mean(valid_scores)
                std_mae = np.std(valid_scores)
                
                results.append({
                    'pipeline': 'ensemble',
                    'model': 'ensemble_average',
                    'mean_mae': mean_mae,
                    'std_mae': std_mae,
                    'n_features': 'multiple',
                    'n_samples': len(df),
                    'n_subjects': len(df['subject_id'].unique()),
                    'successful_folds': len(valid_scores)
                })
                
                print(f"  ENSEMBLE SIMPLE AVERAGE    - Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")

        # 2. Weighted Average Ensemble
        weighted_scores, _ = evaluate_ensemble(results_df, df, ensemble_type='weighted_average')

        if weighted_scores and not all(np.isnan(weighted_scores)):
            # CORRECTED LINE: Check the *individual* score 's' for NaN
            valid_scores = [s for s in weighted_scores if not np.isnan(s)] 
            if valid_scores:
                mean_mae = np.mean(valid_scores)
                std_mae = np.std(valid_scores)
                
                results.append({
                    'pipeline': 'ensemble',
                    'model': 'ensemble_weighted_average',
                    'mean_mae': mean_mae,
                    'std_mae': std_mae,
                    'n_features': 'multiple',
                    'n_samples': len(df),
                    'n_subjects': len(df['subject_id'].unique()),
                    'successful_folds': len(valid_scores)
                })
                
                print(f"  ENSEMBLE WEIGHTED AVERAGE  - Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
        
        results_df = pd.DataFrame(results)
        
    # Print final report
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    all_pipelines_for_report = list(PIPELINES.keys()) + ['ensemble']
    
    if len(results_df) > 0:
        for pipeline in all_pipelines_for_report:
            pipeline_results = results_df[results_df['pipeline'] == pipeline]
            if len(pipeline_results) > 0:
                print(f"\n{pipeline.upper()}:\n")
                pipeline_results = pipeline_results.sort_values(by='mean_mae')
                for _, row in pipeline_results.iterrows():
                    print(f"  {row['model']:25} MAE: {row['mean_mae']:.3f} ± {row['std_mae']:.3f} (folds: {row['successful_folds']})")
        
        print("\n" + "=" * 80)
        print("BEST PERFORMING MODELS")
        print("=" * 80)
        
        successful_results_df = results_df[results_df['successful_folds'] > 0]
        
        if len(successful_results_df) > 0:
            best_overall = successful_results_df.loc[successful_results_df['mean_mae'].idxmin()]
            print(f"Best Overall: {best_overall['pipeline']} with {best_overall['model']}")
            print(f"MAE: {best_overall['mean_mae']:.3f} ± {best_overall['std_mae']:.3f}")
            
            for pipeline in all_pipelines_for_report:
                pipeline_results = successful_results_df[successful_results_df['pipeline'] == pipeline]
                if len(pipeline_results) > 0:
                    pipeline_best = pipeline_results.loc[pipeline_results['mean_mae'].idxmin()]
                    print(f"Best for {pipeline}: {pipeline_best['model']} - MAE: {pipeline_best['mean_mae']:.3f}")
        else:
            print("No models had successful evaluation folds.")
        
        results_df.to_csv('brain_age_results_extended_with_ensembles_upgraded_V7.csv', index=False)
        print(f"\nDetailed results saved to: brain_age_results_extended_with_ensembles_upgraded_V7.csv")
    else:
        print("No successful model evaluations!")
    
    return results_df

if __name__ == "__main__":
    results = main()

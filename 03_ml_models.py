import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configuration
# ----------------------------

# Define the pipelines and their feature sets - UPDATED with correct structure names
PIPELINES = {
    'freesurfer8001ants243': ['freesurfer8001ants243__' + s for s in [
        'Brain-Stem/4thVentricle', 'Left-Thalamus-Proper', 'Right-Thalamus-Proper', 
        'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen', 
        'Left-Pallidum', 'Right-Pallidum', 'Left-Hippocampus', 'Right-Hippocampus', 
        'Left-Amygdala', 'Right-Amygdala', 'Left-Accumbens-area', 'Right-Accumbens-area'
    ]],
    'freesurfer741ants243': ['freesurfer741ants243__' + s for s in [
        'Brain-Stem/4thVentricle', 'Left-Thalamus-Proper', 'Right-Thalamus-Proper', 
        'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen', 
        'Left-Pallidum', 'Right-Pallidum', 'Left-Hippocampus', 'Right-Hippocampus', 
        'Left-Amygdala', 'Right-Amygdala', 'Left-Accumbens-area', 'Right-Accumbens-area'
    ]],
    'samseg8001ants243': ['samseg8001ants243__' + s for s in [
        'Brain-Stem/4thVentricle', 'Left-Thalamus-Proper', 'Right-Thalamus-Proper', 
        'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen', 
        'Left-Pallidum', 'Right-Pallidum', 'Left-Hippocampus', 'Right-Hippocampus', 
        'Left-Amygdala', 'Right-Amygdala', 'Left-Accumbens-area', 'Right-Accumbens-area'
    ]],
    'fslanat6071ants243': ['fslanat6071ants243__' + s for s in [
        'Brain-Stem/4thVentricle', 'Left-Thalamus-Proper', 'Right-Thalamus-Proper', 
        'Left-Caudate', 'Right-Caudate', 'Left-Putamen', 'Right-Putamen', 
        'Left-Pallidum', 'Right-Pallidum', 'Left-Hippocampus', 'Right-Hippocampus', 
        'Left-Amygdala', 'Right-Amygdala', 'Left-Accumbens-area', 'Right-Accumbens-area'
    ]],
    'all_pipelines': None  # Will use all features
}

# Hyperparameter grids - UPDATED: removed Ridge and Lasso, kept only ElasticNet
ELASTIC_NET_PARAMS = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

RANDOM_FOREST_PARAMS = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [5, 10, 15, 20, None],
    'regressor__min_samples_split': [2, 5, 10, 15],
    'regressor__min_samples_leaf': [1, 2, 4]
}

EXTRA_TREES_PARAMS = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [5, 10, 15, 20, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

GRADIENT_BOOSTING_PARAMS = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [2, 3, 4, 5],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__subsample': [0.8, 0.9, 1.0]
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

# ----------------------------
# Data Preparation
# ----------------------------

def clean_age_value(age):
    """Convert age values to numeric, handling special cases like '89+'"""
    if pd.isna(age):
        return np.nan
    if isinstance(age, str):
        # Remove + sign and convert to float
        age = age.replace('+', '').strip()
        # Handle other non-numeric cases
        try:
            return float(age)
        except ValueError:
            return np.nan
    return float(age)

def load_and_preprocess_data(file_path):
    """Load and preprocess the data - NOW WITHOUT NORMALIZATION"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Clean age column - convert to numeric, coercing errors to NaN
    print("Cleaning age data...")
    df['age'] = df['age'].apply(clean_age_value)
    
    # Create unique subject identifier for grouping
    df['subject_id'] = df['dataset'] + '_' + df['subject']
    
    # Remove rows with missing age
    initial_count = len(df)
    df_clean = df.dropna(subset=['age']).copy()
    print(f"After removing missing/invalid age: {len(df_clean)} samples (removed {initial_count - len(df_clean)})")
    
    # For 'all_pipelines', check if any of the volume features are missing
    # Get all feature columns (exclude metadata columns)
    metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id']
    all_feature_cols = [col for col in df_clean.columns if col not in metadata_cols]
    
    initial_count_age_clean = len(df_clean)
    df_clean = df_clean.dropna(subset=all_feature_cols)
    print(f"After removing missing feature data: {len(df_clean)} samples (removed {initial_count_age_clean - len(df_clean)})")
    
    # Verify all feature data is numeric
    for col in all_feature_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Final cleanup - remove any rows that still have NaN in feature columns
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
        # All feature columns (exclude metadata)
        metadata_cols = ['dataset', 'subject', 'session', 'age', 'subject_id']
        return [col for col in df.columns if col not in metadata_cols]
    else:
        # For specific pipelines, we need to find which features actually exist in the dataset
        # that match the pipeline prefix and structure names
        available_features = []
        expected_features = PIPELINES[pipeline_name]
        
        for expected_feature in expected_features:
            # Check if any column starts with this expected feature pattern
            # (since the actual features have additional suffixes like __volume, __surface-area, etc.)
            matching_cols = [col for col in df.columns if col.startswith(expected_feature)]
            available_features.extend(matching_cols)
        
        return available_features

# ----------------------------
# Normalization Functions
# ----------------------------

class DatasetPipelineScaler:
    """
    Custom scaler that computes z-scores per dataset and pipeline combination
    to account for batch effects across datasets and pipelines.
    """
    def __init__(self):
        self.scalers_ = {}  # Store scalers for each (dataset, pipeline) combo
        self.feature_means_ = {}
        self.feature_stds_ = {}
    
    def fit(self, X, features, datasets, pipeline_name):
        """
        Fit scalers for each dataset in the training data
        
        Parameters:
        - X: Feature matrix (numpy array)
        - features: List of feature names
        - datasets: Array of dataset identifiers for each sample
        - pipeline_name: Name of the current pipeline
        """
        unique_datasets = np.unique(datasets)
        
        for dataset in unique_datasets:
            dataset_mask = datasets == dataset
            X_dataset = X[dataset_mask]
            
            # Compute mean and std for this dataset
            if len(X_dataset) > 1:  # Need at least 2 samples for std
                scaler = StandardScaler()
                scaler.fit(X_dataset)
                self.scalers_[(dataset, pipeline_name)] = scaler
                
                # Store means and stds for reference
                self.feature_means_[(dataset, pipeline_name)] = scaler.mean_
                self.feature_stds_[(dataset, pipeline_name)] = scaler.scale_
            else:
                # For single sample, use zero mean and unit variance (or handle differently)
                self.scalers_[(dataset, pipeline_name)] = None
        
        return self
    
    def transform(self, X, features, datasets, pipeline_name):
        """
        Transform data using pre-fitted scalers
        """
        X_normalized = np.zeros_like(X)
        
        unique_datasets = np.unique(datasets)
        
        for dataset in unique_datasets:
            dataset_mask = datasets == dataset
            X_dataset = X[dataset_mask]
            
            if (dataset, pipeline_name) in self.scalers_ and self.scalers_[(dataset, pipeline_name)] is not None:
                # Use pre-fitted scaler
                X_normalized[dataset_mask] = self.scalers_[(dataset, pipeline_name)].transform(X_dataset)
            else:
                # If no scaler available (e.g., new dataset), use identity transformation
                X_normalized[dataset_mask] = X_dataset
        
        return X_normalized

def normalize_features(df, features, pipeline_name, datasets, fit_scaler=None):
    """
    Normalize features using z-score normalization per dataset
    
    Parameters:
    - df: DataFrame containing the features
    - features: List of feature columns to normalize
    - pipeline_name: Name of the current pipeline (for tracking)
    - datasets: Array of dataset identifiers
    - fit_scaler: Optional pre-fitted DatasetPipelineScaler for transformation only
    
    Returns:
    - Normalized DataFrame and fitted scaler
    """
    if len(features) == 0:
        return df, fit_scaler
    
    # Extract feature matrix
    X = df[features].values
    
    if fit_scaler is None:
        # Fit new scaler
        scaler = DatasetPipelineScaler()
        scaler.fit(X, features, datasets, pipeline_name)
        X_normalized = scaler.transform(X, features, datasets, pipeline_name)
    else:
        # Use pre-fitted scaler
        X_normalized = fit_scaler.transform(X, features, datasets, pipeline_name)
        scaler = fit_scaler
    
    # Create normalized DataFrame
    df_normalized = df.copy()
    df_normalized[features] = X_normalized
    
    return df_normalized, scaler

# ----------------------------
# Modeling Functions
# ----------------------------

def create_model_pipeline(model_type):
    """Create a pipeline with the specified model"""
    if model_type == 'elasticnet':
        model = ElasticNet(random_state=42, max_iter=50000)
        param_grid = ELASTIC_NET_PARAMS
    elif model_type == 'randomforest':
        model = RandomForestRegressor(random_state=42)
        param_grid = RANDOM_FOREST_PARAMS
    elif model_type == 'extratrees':
        model = ExtraTreesRegressor(random_state=42)
        param_grid = EXTRA_TREES_PARAMS
    elif model_type == 'gradientboosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = GRADIENT_BOOSTING_PARAMS
    elif model_type == 'svm':
        model = SVR()
        param_grid = SVM_PARAMS
    elif model_type == 'mlp':
        model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
        param_grid = MLP_PARAMS
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline (scaler is now handled separately to avoid data leakage)
    pipeline = Pipeline([
        ('regressor', model)
    ])
    
    return pipeline, param_grid

class EnsembleModel:
    """Simple ensemble that averages predictions from multiple models"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return np.mean(predictions, axis=1)

def nested_cv_evaluation(df, model_type, pipeline_name):
    """Perform nested CV with proper normalization to avoid data leakage"""
    print(f"  Evaluating {model_type} on {pipeline_name}...")
    
    # Get features for this pipeline
    features = get_features_for_pipeline(df, pipeline_name)
    
    if len(features) == 0:
        print(f"    WARNING: No features found for pipeline '{pipeline_name}'")
        return [np.nan], []
    
    # Create model pipeline
    model_pipeline, param_grid = create_model_pipeline(model_type)
    
    # Prepare data
    X = df[features]
    y = df['age']
    groups = df['subject_id']
    datasets = df['dataset']
    
    # Convert to numpy arrays to avoid pandas issues
    X_array = X.values if hasattr(X, 'values') else X
    y_array = y.values if hasattr(y, 'values') else y
    groups_array = groups.values if hasattr(groups, 'values') else groups
    datasets_array = datasets.values if hasattr(datasets, 'values') else datasets
    
    # Outer CV: 5-fold group split for evaluation
    outer_cv = GroupKFold(n_splits=5)
    outer_scores = []
    best_models = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_array, y_array, groups_array)):
        print(f"    Fold {fold + 1}:", end=" ")
        
        # Split data
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        groups_train = groups_array[train_idx]
        datasets_train = datasets_array[train_idx]
        datasets_test = datasets_array[test_idx]
        
        # Create temporary DataFrames for normalization
        train_df = pd.DataFrame(X_train, columns=features)
        train_df['dataset'] = datasets_train
        test_df = pd.DataFrame(X_test, columns=features)
        test_df['dataset'] = datasets_test
        
        # Normalize training data and fit scaler
        train_normalized, scaler = normalize_features(train_df, features, pipeline_name, datasets_train)
        
        # Normalize test data using the scaler fitted on training data
        test_normalized, _ = normalize_features(test_df, features, pipeline_name, datasets_test, fit_scaler=scaler)
        
        # Extract normalized arrays
        X_train_norm = train_normalized[features].values
        X_test_norm = test_normalized[features].values
        
        # Inner CV: 3-fold group split for hyperparameter tuning
        inner_cv = GroupKFold(n_splits=3)
        
        try:
            # Grid search for hyperparameter optimization on normalized training data
            grid_search = GridSearchCV(
                model_pipeline, param_grid, cv=inner_cv, 
                scoring='neg_mean_absolute_error', n_jobs=-1,
                error_score='raise'
            )
            grid_search.fit(X_train_norm, y_train, groups=groups_train)
            
            # Get best model and evaluate on normalized test fold
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_norm)
            mae = mean_absolute_error(y_test, y_pred)
            
            outer_scores.append(mae)
            best_models.append(best_model)
            
            print(f"MAE = {mae:.3f}, Best params: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"FAILED - {str(e)}")
            # If a fold fails, we'll skip it but continue with others
            continue
    
    # If we have at least 3 successful folds, proceed
    if len(outer_scores) >= 3:
        return outer_scores, best_models
    else:
        print(f"  WARNING: Only {len(outer_scores)} successful folds for {model_type}")
        return [np.nan], []

def evaluate_ensemble(results_df, df):
    """Evaluate ensemble model using best individual models from each pipeline"""
    print(f"\n{'='*50}")
    print(f"ENSEMBLE EVALUATION")
    print(f"{'='*50}")
    
    # Find best model for each pipeline
    best_models_info = {}
    for pipeline in PIPELINES.keys():
        if pipeline == 'all_pipelines':  # Skip all_pipelines for ensemble
            continue
            
        pipeline_results = results_df[results_df['pipeline'] == pipeline]
        if len(pipeline_results) > 0:
            best_idx = pipeline_results['mean_mae'].idxmin()
            best_model_info = pipeline_results.loc[best_idx]
            best_models_info[pipeline] = best_model_info
    
    if len(best_models_info) < 2:
        print("Not enough successful pipelines for ensemble")
        return [np.nan], []
    
    print(f"Creating ensemble from {len(best_models_info)} best models:")
    for pipeline, info in best_models_info.items():
        print(f"  {pipeline}: {info['model']} (MAE: {info['mean_mae']:.3f})")
    
    # Prepare data for ensemble
    groups = df['subject_id']
    datasets = df['dataset']
    
    # Outer CV for ensemble evaluation
    outer_cv = GroupKFold(n_splits=5)
    X_full = df.drop(['dataset', 'subject', 'session', 'age', 'subject_id'], axis=1)
    y_array = df['age'].values
    groups_array = groups.values
    datasets_array = datasets.values
    
    outer_scores = []
    ensemble_models = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_array, groups_array)):
        print(f"  Ensemble Fold {fold + 1}:")
        
        X_train_full, X_test_full = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        groups_train = groups_array[train_idx]
        datasets_train = datasets_array[train_idx]
        datasets_test = datasets_array[test_idx]
        
        # Train individual models on training data
        individual_models = []
        
        for pipeline, model_info in best_models_info.items():
            model_type = model_info['model']
            features = get_features_for_pipeline(df.iloc[train_idx], pipeline)
            
            if len(features) == 0:
                continue
                
            # Create temporary DataFrames for this pipeline's features
            train_df_pipeline = pd.DataFrame(X_train_full[features].values, columns=features)
            train_df_pipeline['dataset'] = datasets_train
            
            test_df_pipeline = pd.DataFrame(X_test_full[features].values, columns=features)
            test_df_pipeline['dataset'] = datasets_test
            
            # Normalize training data
            train_normalized, scaler = normalize_features(train_df_pipeline, features, pipeline, datasets_train)
            X_pipeline_train = train_normalized[features].values
            
            # Normalize test data using training scaler
            test_normalized, _ = normalize_features(test_df_pipeline, features, pipeline, datasets_test, fit_scaler=scaler)
            X_pipeline_test = test_normalized[features].values
            
            # Create and train the model
            model_pipeline, param_grid = create_model_pipeline(model_type)
            
            # Simple training without hyperparameter tuning for speed
            try:
                model_pipeline.fit(X_pipeline_train, y_train)
                individual_models.append((model_pipeline, pipeline, features, scaler))
                print(f"    Trained {model_type} for {pipeline}")
            except Exception as e:
                print(f"    Failed to train {model_type} for {pipeline}: {e}")
                continue
        
        if len(individual_models) < 2:
            print(f"    Not enough models trained for ensemble in fold {fold + 1}")
            continue
        
        # Make ensemble prediction (simple average)
        all_predictions = []
        for model, pipeline, features, scaler in individual_models:
            # Create test DataFrame for this pipeline
            test_df_pipeline = pd.DataFrame(X_test_full[features].values, columns=features)
            test_df_pipeline['dataset'] = datasets_test
            
            # Normalize test data using the scaler from training
            test_normalized, _ = normalize_features(test_df_pipeline, features, pipeline, datasets_test, fit_scaler=scaler)
            X_pipeline_test = test_normalized[features].values
            
            pred = model.predict(X_pipeline_test)
            all_predictions.append(pred)
        
        ensemble_pred = np.mean(all_predictions, axis=0)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        outer_scores.append(mae)
        ensemble_models.append(individual_models)
        
        print(f"    Fold {fold + 1}: Ensemble MAE = {mae:.3f}")
    
    if len(outer_scores) >= 3:
        return outer_scores, ensemble_models
    else:
        print(f"  WARNING: Only {len(outer_scores)} successful folds for ensemble")
        return [np.nan], []

# ----------------------------
# Main Execution
# ----------------------------

def main():
    print("=" * 60)
    print("BRAIN AGE PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load data from new file - NOW WITHOUT PRE-NORMALIZATION
    df = load_and_preprocess_data('ml_dataset_with_age.csv')
    
    if len(df) == 0:
        print("ERROR: No valid data remaining after preprocessing!")
        return
    
    # Prepare results storage
    results = []
    
    # Define models to evaluate - UPDATED: removed ridge and lasso, kept only elasticnet
    models_to_evaluate = ['elasticnet', 'randomforest', 'extratrees', 'gradientboosting', 'svm', 'mlp']
    
    # Iterate through each pipeline configuration
    for pipeline_name in PIPELINES.keys():
        print(f"\n{'='*50}")
        print(f"ANALYZING: {pipeline_name.upper()}")
        print(f"{'='*50}")
        
        # Get features for this pipeline
        features = get_features_for_pipeline(df, pipeline_name)
        
        # Check if we found any features for this pipeline
        if len(features) == 0:
            print(f"  WARNING: No features found for pipeline '{pipeline_name}'")
            continue
            
        print(f"Features: {len(features)}, Samples: {len(df)}, Subjects: {len(df['subject_id'].unique())}")
        
        # Evaluate all model types with proper nested CV normalization
        for model_type in models_to_evaluate:
            # Perform nested CV with proper normalization
            scores, models = nested_cv_evaluation(df, model_type, pipeline_name)
            
            # Only store results if we have valid scores
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
                    
                    print(f"  {model_type.upper()} - Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
                else:
                    print(f"  {model_type.upper()} - No valid results")
            else:
                print(f"  {model_type.upper()} - Evaluation failed")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Evaluate ensemble model if we have results
    if len(results_df) > 0:
        ensemble_scores, ensemble_models = evaluate_ensemble(results_df, df)
        
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
                
                print(f"  ENSEMBLE - Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
        
        # Update results dataframe with ensemble
        results_df = pd.DataFrame(results)
    
    # Print final report
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    for pipeline in list(PIPELINES.keys()) + ['ensemble']:
        pipeline_results = results_df[results_df['pipeline'] == pipeline]
        if len(pipeline_results) > 0:
            print(f"\n{pipeline.upper()}:")
            for _, row in pipeline_results.iterrows():
                print(f"  {row['model']:20} MAE: {row['mean_mae']:.3f} ± {row['std_mae']:.3f} (folds: {row['successful_folds']})")
    
    # Find best performing models
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODELS")
    print("=" * 80)
    
    if len(results_df) > 0:
        best_overall = results_df.loc[results_df['mean_mae'].idxmin()]
        print(f"Best Overall: {best_overall['pipeline']} with {best_overall['model']}")
        print(f"MAE: {best_overall['mean_mae']:.3f} ± {best_overall['std_mae']:.3f}")
        
        # Best per pipeline
        for pipeline in list(PIPELINES.keys()) + ['ensemble']:
            pipeline_results = results_df[results_df['pipeline'] == pipeline]
            if len(pipeline_results) > 0:
                pipeline_best = pipeline_results.loc[pipeline_results['mean_mae'].idxmin()]
                print(f"Best for {pipeline}: {pipeline_best['model']} - MAE: {pipeline_best['mean_mae']:.3f}")
    else:
        print("No successful model evaluations!")
    
    # Save detailed results
    if len(results_df) > 0:
        results_df.to_csv('brain_age_results.csv', index=False)
        print(f"\nDetailed results saved to: brain_age_results.csv")
    else:
        print("\nNo results to save!")
    
    return results_df

if __name__ == "__main__":
    results = main()

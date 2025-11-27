import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('brain_age_results_sex_split_and_correct_ensembles.csv')

# Clean up model names by removing 'ensemble_' prefix
df['model_clean'] = df['model'].str.replace('ensemble_', '')

# Create simplified pipeline names with updated labels
def simplify_pipeline(name):
    if 'freesurfer8001ants243' in name:
        return 'FS8001'
    elif 'freesurfer741ants243' in name:
        return 'FS741'
    elif 'samseg8001ants243' in name:
        return 'SAMSEG8'
    elif 'fslanat6071ants243' in name:
        return 'FSL6071'
    elif 'all_pipelines' in name:
        return 'Data Augmentation'
    elif 'ensemble' in name:
        return 'Ensemble'
    else:
        return name

df['pipeline_simple'] = df['pipeline'].apply(simplify_pipeline)

# Set up the plotting style
sns.set_style("whitegrid")

# Create side-by-side plots for Male and Female
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))

# Get n_subjects for titles
male_n = df[df['sex'] == 'M']['n_subjects'].iloc[0]
female_n = df[df['sex'] == 'F']['n_subjects'].iloc[0]

# Define better colors for pipelines (darker colors for better text contrast)
pipelines = df['pipeline_simple'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # Better color palette
color_dict = dict(zip(pipelines, colors))

# Male subjects - manual plotting
male_data = df[df['sex'] == 'M']
models = male_data['model_clean'].unique()

bar_width = 0.8  # Thick bars
for model_idx, model in enumerate(models):
    model_data = male_data[male_data['model_clean'] == model]
    
    # Special handling for baseline - use only one bar
    if model == 'baseline':
        row = model_data.iloc[0]  # Take the first baseline row (all are identical)
        x_pos = model_idx
        
        # Plot bar
        bar = ax1.bar(x_pos, row['mean_mae'], width=bar_width, 
                     color='#7f7f7f', alpha=0.8)  # Gray for baseline
        
        # Error bars - properly positioned
        ax1.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                    color='black', capsize=4, capthick=1, linewidth=1.5)
        
        # Pipeline label on bars - VERTICAL TEXT with better contrast
        ax1.text(x_pos, row['mean_mae']/2, 'Baseline',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                rotation=90)
    
    else:
        n_bars = len(model_data)
        for pipeline_idx, (_, row) in enumerate(model_data.iterrows()):
            x_pos = model_idx + (pipeline_idx - n_bars/2 + 0.5) * (bar_width/n_bars)
            
            # Plot bar
            bar = ax1.bar(x_pos, row['mean_mae'], width=bar_width/n_bars, 
                         color=color_dict[row['pipeline_simple']], alpha=0.8)
            
            # Error bars - properly positioned
            ax1.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                        color='black', capsize=4, capthick=1, linewidth=1.5)
            
            # Pipeline label on bars - VERTICAL TEXT with better contrast
            ax1.text(x_pos, row['mean_mae']/2, row['pipeline_simple'],
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                    rotation=90)

ax1.set_title(f'Male Subjects - Brain Age Prediction MAE (n={int(male_n)})', fontsize=14, fontweight='bold')
ax1.set_xlabel('Model Type', fontsize=12)
ax1.set_ylabel('Mean MAE (years) ± STD', fontsize=12)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=0)  # HORIZONTAL LABELS
ax1.set_xlim(-0.5, len(models) - 0.5)

# Female subjects - manual plotting
female_data = df[df['sex'] == 'F']
for model_idx, model in enumerate(models):
    model_data = female_data[female_data['model_clean'] == model]
    
    # Special handling for baseline - use only one bar
    if model == 'baseline':
        row = model_data.iloc[0]  # Take the first baseline row (all are identical)
        x_pos = model_idx
        
        # Plot bar
        bar = ax2.bar(x_pos, row['mean_mae'], width=bar_width,
                     color='#7f7f7f', alpha=0.8)  # Gray for baseline
        
        # Error bars - properly positioned
        ax2.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                    color='black', capsize=4, capthick=1, linewidth=1.5)
        
        # Pipeline label on bars - VERTICAL TEXT with better contrast
        ax2.text(x_pos, row['mean_mae']/2, 'Baseline',
                ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                rotation=90)
    
    else:
        n_bars = len(model_data)
        for pipeline_idx, (_, row) in enumerate(model_data.iterrows()):
            x_pos = model_idx + (pipeline_idx - n_bars/2 + 0.5) * (bar_width/n_bars)
            
            # Plot bar
            bar = ax2.bar(x_pos, row['mean_mae'], width=bar_width/n_bars,
                         color=color_dict[row['pipeline_simple']], alpha=0.8)
            
            # Error bars - properly positioned
            ax2.errorbar(x_pos, row['mean_mae'], yerr=row['std_mae'],
                        color='black', capsize=4, capthick=1, linewidth=1.5)
            
            # Pipeline label on bars - VERTICAL TEXT with better contrast
            ax2.text(x_pos, row['mean_mae']/2, row['pipeline_simple'],
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white',
                    rotation=90)

ax2.set_title(f'Female Subjects - Brain Age Prediction MAE (n={int(female_n)})', fontsize=14, fontweight='bold')
ax2.set_xlabel('Model Type', fontsize=12)
ax2.set_ylabel('Mean MAE (years) ± STD', fontsize=12)
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=0)  # HORIZONTAL LABELS
ax2.set_xlim(-0.5, len(models) - 0.5)

plt.tight_layout()

# Save as PNG
plt.savefig('brain_age_side_by_side_sexes.png', dpi=300, bbox_inches='tight')
plt.close()

print("PNG image saved: brain_age_side_by_side_sexes.png")

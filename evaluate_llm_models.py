"""
LIWC Features Classification Pipeline
Evaluates LIWC features on binary classification task using 5-fold cross-validation.
Includes SHAP analysis for feature importance explanation.
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# SHAP imports
import shap
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)


class LIWCClassifier:
    """LIWC features classifier with 5-fold cross-validation."""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation on LIWC features."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("LIWC Features - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            start_time = time.time()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train classifier
            self.classifier.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_times


def load_liwc_data(filepath):
    """Load and preprocess LIWC features dataset."""
    print(f"\nLoading LIWC dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"LIWC dataset shape: {df.shape}")
    
    # Metadata columns to exclude
    metadata_cols = ['pageid', 'title', 'content', 'categories', 'is_ai_flagged', 'Segment', 'WC']
    
    # Get LIWC feature columns (all columns except metadata)
    liwc_feature_cols = [col for col in df.columns if col not in metadata_cols]
    print(f"Number of LIWC features: {len(liwc_feature_cols)}")
    
    # Check for missing values
    missing_values = df[liwc_feature_cols].isna().sum().sum()
    print(f"Missing values in LIWC features: {missing_values}")
    
    # Handle missing values
    df = df.dropna(subset=liwc_feature_cols)
    
    # Check target variable
    print(f"\nTarget variable 'is_ai_flagged' distribution:")
    print(df['is_ai_flagged'].value_counts())
    print(f"Class balance: {df['is_ai_flagged'].value_counts(normalize=True)}")
    
    # Extract features and labels
    X = df[liwc_feature_cols].values.astype(np.float32)
    y = df['is_ai_flagged'].values
    
    # Use full dataset (no sample limit)
    # print(f"\nLimiting to 10 samples for testing...")
    # X = X[:10]
    # y = y[:10]
    
    print(f"\nFinal LIWC dataset size: {len(X)} samples")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, liwc_feature_cols


def print_results_summary(liwc_results):
    """Print formatted results summary."""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # LIWC Results
    if liwc_results is not None:
        liwc_accs, liwc_times = liwc_results
        print("LIWC Features:")
        print(f"  Mean Accuracy: {np.mean(liwc_accs):.4f} ± {np.std(liwc_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in liwc_accs]}")
        print(f"  Mean Time per Fold: {np.mean(liwc_times):.2f}s ± {np.std(liwc_times):.2f}s")
        print(f"  Total Time: {np.sum(liwc_times):.2f}s")
    
    print(f"\n{'='*80}\n")


def save_results_to_csv(liwc_results, output_file="evaluation_results.csv"):
    """Save results to CSV file."""
    # Create base DataFrame
    results_dict = {
        'Fold': range(1, 6)
    }
    
    # Add LIWC results if available
    if liwc_results is not None:
        liwc_accs, liwc_times = liwc_results
        results_dict['LIWC_Accuracy'] = liwc_accs
        results_dict['LIWC_Time'] = liwc_times
    
    results_df = pd.DataFrame(results_dict)
    
    # Add summary row
    summary_dict = {
        'Fold': ['Mean', 'Std']
    }
    
    if liwc_results is not None:
        liwc_accs, liwc_times = liwc_results
        summary_dict['LIWC_Accuracy'] = [np.mean(liwc_accs), np.std(liwc_accs)]
        summary_dict['LIWC_Time'] = [np.mean(liwc_times), np.std(liwc_times)]
    
    summary_row = pd.DataFrame(summary_dict)
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def perform_shap_analysis(X, y, feature_names, output_dir="shap_results", n_samples=1000):
    """
    Perform SHAP analysis on LIWC features.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save SHAP results
    n_samples : int
        Number of samples to use for SHAP analysis (for computational efficiency)
    """
    print(f"\n{'='*80}")
    print("SHAP ANALYSIS")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data for training and SHAP analysis
    print("Preparing data for SHAP analysis...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a model for SHAP analysis
    print("Training model for SHAP analysis...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Model Accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    # Sample data for SHAP (to reduce computation time)
    # Use background data (training set) and explanation data (test set)
    n_background = min(100, len(X_train_scaled))
    n_explain = min(n_samples, len(X_test_scaled))
    
    X_background = X_train_scaled[:n_background]
    X_explain = X_test_scaled[:n_explain]
    
    print(f"\nComputing SHAP values...")
    print(f"  Background samples: {n_background}")
    print(f"  Explanation samples: {n_explain}")
    
    # Create SHAP explainer (TreeExplainer is efficient for RandomForest)
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_explain)
    
    # For binary classification, shap_values can be:
    # - A list [class_0_values, class_1_values] where each is 2D (n_samples, n_features)
    # - A 3D array (n_samples, n_features, n_classes)
    # We'll use class_1 (positive class) for analysis
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]  # Use positive class
    else:
        shap_values_class1 = shap_values
    
    # Handle 3D array case: (n_samples, n_features, n_classes)
    # Extract class 1 (index 1) from the last dimension
    if len(shap_values_class1.shape) == 3:
        print(f"  Note: shap_values_class1 is 3D with shape {shap_values_class1.shape}, extracting class 1")
        shap_values_class1 = shap_values_class1[:, :, 1]  # Select class 1 (positive class)
    
    print("SHAP values computed successfully!")
    print(f"  Final shap_values_class1 shape: {shap_values_class1.shape}")
    
    # 1. Summary plot (bar plot of mean absolute SHAP values)
    print("\nGenerating SHAP summary plots...")
    plt.figure(figsize=(18, len(feature_names) * 0.3)) # Increase height for all features
    shap.summary_plot(shap_values_class1, X_explain, feature_names=feature_names, 
                 plot_type="bar", show=False, max_display=len(feature_names)) # Use all features
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/shap_summary_bar.png")
    
    # 2. Summary plot (beeswarm plot)
    plt.figure(figsize=(18, len(feature_names) * 0.3)) # Increase height for all features
    shap.summary_plot(shap_values_class1, X_explain, feature_names=feature_names, 
                 plot_type="dot", show=False, max_display=len(feature_names)) # Use all features
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/shap_summary_beeswarm.png")
    
    # 3. Feature importance ranking
    # Ensure shap_values_class1 is 2D: (n_samples, n_features)
    if len(shap_values_class1.shape) == 1:
        shap_values_class1 = shap_values_class1.reshape(-1, len(feature_names))
    elif len(shap_values_class1.shape) == 3:
        # If still 3D, extract class 1
        shap_values_class1 = shap_values_class1[:, :, 1]
    
    # Verify shape is now 2D: (n_samples, n_features)
    if len(shap_values_class1.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {shap_values_class1.shape}")
    
    # Check if shape matches expected (n_samples, n_features)
    if shap_values_class1.shape[1] != len(feature_names):
        # If first dimension matches feature count, it's likely transposed
        if shap_values_class1.shape[0] == len(feature_names):
            print(f"  Warning: Transposing shap_values_class1 from {shap_values_class1.shape} to match features")
            shap_values_class1 = shap_values_class1.T
        else:
            raise ValueError(
                f"Shape mismatch: shap_values_class1 has {shap_values_class1.shape[1]} features "
                f"but expected {len(feature_names)} features. "
                f"Array shape: {shap_values_class1.shape}"
            )
    
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    std_shap = shap_values_class1.std(axis=0)
    
    # Explicitly ensure arrays are 1D (flatten if needed)
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    std_shap = np.asarray(std_shap).flatten()
    
    # Verify lengths match
    if len(mean_abs_shap) != len(feature_names):
        raise ValueError(
            f"Length mismatch: mean_abs_shap has {len(mean_abs_shap)} elements, "
            f"but feature_names has {len(feature_names)} features. "
            f"shap_values_class1 shape: {shap_values_class1.shape}"
        )
    if len(std_shap) != len(feature_names):
        raise ValueError(
            f"Length mismatch: std_shap has {len(std_shap)} elements, "
            f"but feature_names has {len(feature_names)} features"
        )
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Std_SHAP': std_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    feature_importance_df.to_csv(
        os.path.join(output_dir, "shap_feature_importance.csv"), 
        index=False
    )
    print(f"  Saved: {output_dir}/shap_feature_importance.csv")
    
   
    print(f"\nAll {len(feature_names)} LIWC Features Ranked by SHAP:")
    print("-" * 80)
    # Loop over the entire sorted DataFrame (using len(feature_importance_df))
    for idx, row in feature_importance_df.iterrows():
      print(f"{row['Feature']:30s} | Mean |SHAP|: {row['Mean_Abs_SHAP']:8.4f}")
    
    # 4. Save SHAP values for all samples
    shap_values_df = pd.DataFrame(
        shap_values_class1,
        columns=feature_names
    )
    shap_values_df.to_csv(
        os.path.join(output_dir, "shap_values.csv"),
        index=False
    )
    print(f"\n  Saved: {output_dir}/shap_values.csv")
    
    # 5. Waterfall plot for a single example (first instance)
    print("\nGenerating example waterfall plot...")
    try:
        # Extract base value properly (handle array/list cases)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            if len(explainer.expected_value) > 1:
                base_value = float(explainer.expected_value[1])  # Class 1 (positive class)
            else:
                base_value = float(explainer.expected_value[0])
        else:
            base_value = float(explainer.expected_value)
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_class1[0],
                base_values=base_value,
                data=X_explain[0],
                feature_names=feature_names
            ),
            show=False,
            max_display=20
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_waterfall_example.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/shap_waterfall_example.png")
    except Exception as e:
        print(f"  Warning: Could not generate waterfall plot: {e}")
        print("  Skipping waterfall plot...")
    
    # 6. Dependence plots for top features
    print("\nGenerating dependence plots for top 5 features...")
    top_features = feature_importance_df.head(5)['Feature'].tolist()
    top_feature_indices = [feature_names.index(f) for f in top_features]
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (feature, idx) in enumerate(zip(top_features[:5], top_feature_indices[:5])):
            # Find the feature with highest interaction
            interaction_idx = np.abs(shap_values_class1[:, idx]).argmax()
            shap.dependence_plot(
                idx, shap_values_class1, X_explain,
                feature_names=feature_names,
                interaction_index='auto',
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f'Dependence: {feature}', fontsize=10)
        
        # Remove extra subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_dependence_plots.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/shap_dependence_plots.png")
    except Exception as e:
        print(f"  Warning: Could not generate dependence plots: {e}")
        print("  Skipping dependence plots...")
    
    print(f"\n{'='*80}")
    print("SHAP Analysis Complete!")
    print(f"All results saved to: {output_dir}/")
    print(f"{'='*80}\n")
    
    return feature_importance_df


def main():
    """Main evaluation pipeline."""
    liwc_path = "LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv"
    narrative_path = "LIWC-22 Results - combined_training_dataset - Narrative Arc Table.csv"
    
    # Load LIWC data
    X_liwc, y_liwc, feature_names = load_liwc_data(liwc_path)
    
    # Initialize LIWC classifier
    liwc_classifier = LIWCClassifier()
    
    # Evaluate LIWC
    print("\n" + "="*80)
    print("STARTING LIWC FEATURES EVALUATION")
    print("="*80)
    liwc_results = liwc_classifier.evaluate_cv(X_liwc, y_liwc, n_splits=5)
    
    # Print summary
    print_results_summary(liwc_results)
    
    # Save results
    save_results_to_csv(liwc_results)
    
    # Perform SHAP analysis
    print("\n" + "="*80)
    print("STARTING SHAP ANALYSIS")
    print("="*80)
    shap_importance = perform_shap_analysis(
        X_liwc, y_liwc, feature_names, 
        output_dir="shap_results",
        n_samples=1000  # Adjust based on computational resources
    )
    
    print("Evaluation and SHAP analysis complete!")


if __name__ == "__main__":
    main()


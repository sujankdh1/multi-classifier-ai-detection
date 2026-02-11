"""
LIWC Features Classification Pipeline with SHAP Analysis
Evaluates LIWC features using Random Forest with cross-validation and SHAP interpretability.
"""

import pandas as pd
import numpy as np
import time
import warnings
import os

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# SHAP for model interpretability
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)


class LIWCClassifier:
    """LIWC features classifier with 5-fold cross-validation and SHAP support."""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.final_model = None  # Store final trained model for SHAP
        self.final_scaler = None  # Store final scaler for SHAP
        self.feature_names = None  # Store feature names for SHAP
        
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation on LIWC features."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_f1_scores = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("LIWC Features - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        print(f"Number of LIWC features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            start_time = time.time()
            
            # Scale features
            print("  Scaling features...", flush=True)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train classifier
            print("  Training Random Forest...", flush=True)
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            print("  Making predictions...", flush=True)
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='binary')
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_f1_scores, fold_times
    
    def train_final_model(self, X, y, feature_names=None):
        """
        Train a final model on all data for SHAP analysis.
        
        Args:
            X: Array of LIWC features
            y: Array of labels
            feature_names: List of feature names
        
        Returns:
            Tuple of (X_scaled, y) where X_scaled is the transformed feature matrix
        """
        print("\n" + "="*60)
        print("Training Final Model for SHAP Analysis")
        print("="*60)
        
        # Store feature names
        self.feature_names = feature_names if feature_names is not None else np.array([f"feature_{i}" for i in range(X.shape[1])])
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        print("Training RandomForest classifier...")
        self.classifier.fit(X_scaled, y)
        
        # Store final components for SHAP
        self.final_model = self.classifier
        self.final_scaler = self.scaler
        
        print(f"Final model trained on {len(X)} samples with {X_scaled.shape[1]} features.")
        
        return X_scaled, y


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
    
    print(f"\nFinal LIWC dataset size: {len(X)} samples")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, liwc_feature_cols


def print_liwc_results_summary(liwc_results):
    """Print formatted results summary for LIWC features."""
    print(f"\n{'='*80}")
    print("LIWC FEATURES EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    liwc_accs, liwc_f1s, liwc_times = liwc_results
    print("LIWC Features with Random Forest:")
    print(f"  Mean Accuracy: {np.mean(liwc_accs):.4f} ± {np.std(liwc_accs):.4f}")
    print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in liwc_accs]}")
    print(f"  Mean F1 Score: {np.mean(liwc_f1s):.4f} ± {np.std(liwc_f1s):.4f}")
    print(f"  Per-fold F1 Scores: {[f'{f1:.4f}' for f1 in liwc_f1s]}")
    print(f"  Mean Time per Fold: {np.mean(liwc_times):.2f}s ± {np.std(liwc_times):.2f}s")
    print(f"  Total Time: {np.sum(liwc_times):.2f}s")
    
    print(f"\n{'='*80}\n")


def save_liwc_results_to_csv(liwc_results, output_file="liwc_evaluation_results.csv"):
    """Save LIWC evaluation results to CSV file."""
    liwc_accs, liwc_f1s, liwc_times = liwc_results
    
    results_dict = {
        'Fold': list(range(1, 6)),
        'LIWC_Accuracy': liwc_accs,
        'LIWC_F1_Score': liwc_f1s,
        'LIWC_Time': liwc_times
    }
    
    results_df = pd.DataFrame(results_dict)
    
    # Add summary row
    summary_dict = {
        'Fold': ['Mean', 'Std'],
        'LIWC_Accuracy': [np.mean(liwc_accs), np.std(liwc_accs)],
        'LIWC_F1_Score': [np.mean(liwc_f1s), np.std(liwc_f1s)],
        'LIWC_Time': [np.mean(liwc_times), np.std(liwc_times)]
    }
    
    summary_row = pd.DataFrame(summary_dict)
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def generate_shap_explanations(classifier, X_sample, y_sample=None, max_samples=1000, output_dir="shap_results"):
    """
    Generate SHAP explanations for a trained classifier.
    
    Args:
        classifier: Trained classifier object (must have final_model, final_scaler, and feature_names attributes)
        X_sample: Sample data to explain (transformed/scaled features)
        y_sample: Optional labels for the sample data
        max_samples: Maximum number of samples to use for SHAP (for performance)
        output_dir: Directory to save SHAP visualizations and results
    
    Returns:
        shap_values: SHAP values array
        shap_explainer: SHAP explainer object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating SHAP Explanations")
    print("="*60)
    
    # Check if classifier has required attributes
    if not hasattr(classifier, 'final_model') or classifier.final_model is None:
        raise ValueError("Classifier must have a trained final_model. Call train_final_model() first.")
    
    # Limit sample size for performance
    if len(X_sample) > max_samples:
        print(f"Limiting SHAP analysis to {max_samples} samples (from {len(X_sample)} total)")
        # Use stratified sampling if labels are available
        if y_sample is not None:
            from sklearn.model_selection import train_test_split
            X_sample, _, y_sample, _ = train_test_split(
                X_sample, y_sample, 
                train_size=max_samples, 
                stratify=y_sample, 
                random_state=42
            )
        else:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]
            if y_sample is not None:
                y_sample = y_sample[indices]
    
    # Verify we got the right number of samples
    actual_samples = len(X_sample)
    print(f"Using {actual_samples} samples for SHAP analysis...")
    
    # Create SHAP explainer (TreeExplainer for RandomForest)
    print("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(classifier.final_model)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_sample)
    
    # Handle binary classification (SHAP returns list for binary)
    if isinstance(shap_values, list):
        # For binary classification, use the positive class (index 1)
        shap_values = shap_values[1]
    
    print(f"SHAP values calculated. Shape: {shap_values.shape}")
    
    # Save SHAP values to CSV
    if classifier.feature_names is not None:
        shap_df = pd.DataFrame(
            shap_values,
            columns=[f"SHAP_{name}" for name in classifier.feature_names]
        )
        shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
        print(f"SHAP values saved to {output_dir}/shap_values.csv")
    
    # Generate and save visualizations
    print("\nGenerating SHAP visualizations...")
    
    # 1. Summary plot (bar plot of mean absolute SHAP values)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, 
                         feature_names=classifier.feature_names,
                         show=False, plot_type="bar", max_display=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary bar plot saved to {output_dir}/shap_summary_bar.png")
    except Exception as e:
        print(f"  ✗ Error creating summary bar plot: {e}")
    
    # 2. Summary plot (beeswarm plot)
    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=classifier.feature_names,
                         show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary beeswarm plot saved to {output_dir}/shap_summary_beeswarm.png")
    except Exception as e:
        print(f"  ✗ Error creating summary beeswarm plot: {e}")
    
    # 3. Feature importance (mean absolute SHAP values)
    try:
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[::-1][:30]
        
        if classifier.feature_names is not None:
            top_features = classifier.feature_names[top_indices]
            top_importance = feature_importance[top_indices]
            
            importance_df = pd.DataFrame({
                'Feature': top_features,
                'Mean_Abs_SHAP_Value': top_importance
            })
            importance_df.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)
            print(f"  ✓ Feature importance saved to {output_dir}/shap_feature_importance.csv")
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top 30 LIWC Features by Mean Absolute SHAP Value')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Feature importance plot saved to {output_dir}/shap_feature_importance.png")
    except Exception as e:
        print(f"  ✗ Error creating feature importance: {e}")
    
    # 4. Waterfall plot for a few example instances
    try:
        n_examples = min(5, len(X_sample))
        # Get expected value (handle binary classification)
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
        else:
            base_value = explainer.expected_value
        
        for i in range(n_examples):
            # Create Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values[i],
                base_values=base_value,
                data=X_sample[i],
                feature_names=classifier.feature_names
            )
            
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(explanation, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_waterfall_example_{i+1}.png"), dpi=150, bbox_inches='tight')
            plt.close()
        print(f"  ✓ Waterfall plots saved for {n_examples} examples")
    except Exception as e:
        print(f"  ✗ Error creating waterfall plots: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nSHAP analysis complete! Results saved to {output_dir}/")
    
    return shap_values, explainer


def main():
    """Main evaluation pipeline."""
    liwc_path = "LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv"
    
    # Check if file exists
    if not os.path.exists(liwc_path):
        print(f"Error: File not found: {liwc_path}")
        return
    
    # Load LIWC data
    X, y, feature_names = load_liwc_data(liwc_path)
    
    # Initialize LIWC classifier
    liwc_classifier = LIWCClassifier()
    
    # Perform cross-validation evaluation
    print("\n" + "="*80)
    print("STARTING LIWC FEATURES EVALUATION")
    print("="*80)
    liwc_results = liwc_classifier.evaluate_cv(X, y, n_splits=5)
    
    # Print summary
    print_liwc_results_summary(liwc_results)
    
    # Save results
    save_liwc_results_to_csv(liwc_results)
    
    # Generate SHAP explanations
    print("\n" + "="*80)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*80)
    try:
        # Train final model on all data for SHAP
        X_scaled, y_processed = liwc_classifier.train_final_model(X, y, feature_names=feature_names)
        
        # Generate SHAP explanations
        shap_values, shap_explainer = generate_shap_explanations(
            liwc_classifier,
            X_scaled,
            y_processed, 
            max_samples=1000,
            output_dir="shap_results"
        )
        
        print("\n✓ SHAP analysis completed successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Warning: SHAP analysis failed: {e}")
        print("Continuing without SHAP explanations...")
        import traceback
        traceback.print_exc()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

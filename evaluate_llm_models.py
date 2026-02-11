"""
LIWC Features Classification Pipeline with F1 Score Evaluation
"""

import pandas as pd
import numpy as np
import warnings
import time
import os

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')
np.random.seed(42)

def load_liwc_data(filepath):
    df = pd.read_csv(filepath)
    metadata_cols = ['pageid', 'title', 'content', 'categories', 'is_ai_flagged', 'Segment', 'WC']
    liwc_feature_cols = [col for col in df.columns if col not in metadata_cols]
    df = df.dropna(subset=liwc_feature_cols)
    X = df[liwc_feature_cols].values.astype(np.float32)
    y = df['is_ai_flagged'].values
    return X, y, liwc_feature_cols

def evaluate_liwc_cv(X, y, n_splits=5):
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

def main():
    path = "LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv"
    if not os.path.exists(path):
        print("File not found.")
        return

    X, y, names = load_liwc_data(path)
    
    # Perform cross-validation evaluation
    print("\n" + "="*80)
    print("STARTING LIWC FEATURES EVALUATION")
    print("="*80)
    liwc_results = evaluate_liwc_cv(X, y, n_splits=5)
    
    # Print summary
    print_liwc_results_summary(liwc_results)
    
    # Save results
    save_liwc_results_to_csv(liwc_results)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
"""
LIWC Features Classification Pipeline - Unified Multi-View
Generates Top 20, 30, and 50 plots with a consistent feature order.
"""

import pandas as pd
import numpy as np
import warnings
import os
from scipy.stats import pearsonr

# Scikit-learn and SHAP imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

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

def get_unified_ranking(shap_values, X_data, feature_names):
    """
    Creates a single master list of features ranked by separation strength.
    This ensures Top 20 is a subset of Top 30, etc.
    """
    rankings = []
    for i in range(X_data.shape[1]):
        if np.std(X_data[:, i]) > 0 and np.std(shap_values[:, i]) > 0:
            # We calculate the 'Cleanness' of the feature
            corr, _ = pearsonr(X_data[:, i], shap_values[:, i])
            # We combine Importance (Mean Abs SHAP) with Separation (Correlation)
            importance = np.abs(shap_values[:, i]).mean()
            separation_score = abs(corr) * importance 
            rankings.append((i, separation_score))
        else:
            rankings.append((i, 0))
    
    # Sort by the combined score
    rankings.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in rankings]

def generate_plots(master_indices, shap_values, X_data, feature_names, output_dir):
    """Generates three consistent plots based on the master ranking."""
    for n in [20, 30, 50]:
        target_indices = master_indices[:n]
        
        filtered_shap = shap_values[:, target_indices]
        filtered_X = X_data[:, target_indices]
        filtered_names = [feature_names[i] for i in target_indices]
        
        # Scale height based on number of features
        plt.figure(figsize=(14, n * 0.4 + 2)) 
        
        shap.summary_plot(
            filtered_shap, 
            filtered_X, 
            feature_names=filtered_names, 
            plot_type="dot", 
            show=False, 
            max_display=n
        )
        
        plt.title(f"Consistent Ranking: Top {n} Separated Features", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_consistent_top_{n}.png"), dpi=300)
        plt.close()
        print(f"  Saved Consistent Top {n}")

def main():
    path = "LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv"
    if not os.path.exists(path):
        print("File not found.")
        return

    X, y, names = load_liwc_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test) # Simple scaling for explanation
    
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.fit_transform(X_train), y_train)
    
    explainer = shap.TreeExplainer(model)
    # Correctly handle class output
    raw_shap = explainer.shap_values(X_test_scaled)
    shap_v = raw_shap[1] if isinstance(raw_shap, list) else (raw_shap[:,:,1] if len(raw_shap.shape)==3 else raw_shap)

    os.makedirs("shap_results", exist_ok=True)
    
    # Get one ranking to rule them all
    master_idx = get_unified_ranking(shap_v, X_test_scaled, names)
    
    generate_plots(master_idx, shap_v, X_test_scaled, names, "shap_results")
    print("\nDone! All three images now have the exact same feature order.")

if __name__ == "__main__":
    main()
"""
LLM Text Classification Pipeline
Evaluates Universal Sentence Encoder (USE) and BERT models on binary classification task
using 5-fold cross-validation.
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
import subprocess
import sys
warnings.filterwarnings('ignore')

# NLTK for sentence tokenization
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# TensorFlow and USE imports - MOVED TO LAZY IMPORT
# TensorFlow will only be imported when USE model is actually loaded
# This avoids random_device errors on systems where TensorFlow has issues
import os
# Set environment variables early (before any TensorFlow import attempt)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Transformers and PyTorch imports
from transformers import AutoTokenizer, AutoModel
import torch

# Set random seeds for reproducibility
np.random.seed(42)
# tf.random.set_seed(42)
torch.manual_seed(42)


class TFIDFClassifier:
    """TF-IDF features classifier with 5-fold cross-validation."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize TF-IDF classifier.
        
        Args:
            max_features: Maximum number of features to extract (default: 5000)
            ngram_range: Range of n-grams to use (default: (1, 2) for unigrams and bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        self.classifier = LinearSVC(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation on TF-IDF features."""
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_f1_scores = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("TF-IDF Features - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        print(f"Max Features: {self.max_features}, N-gram Range: {self.ngram_range}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Extract TF-IDF features
            start_time = time.time()
            X_train_tfidf = self.vectorizer.fit_transform(X_train.tolist())
            X_val_tfidf = self.vectorizer.transform(X_val.tolist())
            
            # Convert sparse matrix to dense array for scaling
            X_train_tfidf = X_train_tfidf.toarray()
            X_val_tfidf = X_val_tfidf.toarray()
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_tfidf)
            X_val_scaled = self.scaler.transform(X_val_tfidf)
            
            # Train classifier
            self.classifier.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='binary')
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_f1_scores.append(f1)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_f1_scores, fold_times


def load_and_preprocess_data_full_text(filepath):
    """Load and preprocess the dataset without extracting first sentence (for TF-IDF)."""
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values in content column
    missing_content = df['content'].isna().sum()
    print(f"Missing values in 'content': {missing_content}")
    
    # Handle missing values
    df = df.dropna(subset=['content'])
    
    # Check target variable
    print(f"\nTarget variable 'is_ai_flagged' distribution:")
    print(df['is_ai_flagged'].value_counts())
    print(f"Class balance: {df['is_ai_flagged'].value_counts(normalize=True)}")
    
    # Prepare data
    X = df['content'].values
    y = df['is_ai_flagged'].values
    
    # Convert text to string if needed
    X = [str(text) for text in X]
    
    # Keep full text (no first sentence extraction for TF-IDF)
    print("\nUsing full text for TF-IDF (no sentence extraction)...")
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(X) if text.strip()]
    X = [X[i] for i in valid_indices]
    y = [y[i] for i in valid_indices]
    
    print(f"\nFinal dataset size: {len(X)} samples")
    print(f"Sample full text (first 100 chars): {X[0][:100]}..." if X else "No samples")
    
    return X, y


def print_results_summary(tfidf_results=None):
    """Print formatted results summary."""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # TF-IDF Results
    if tfidf_results is not None:
        tfidf_accs, tfidf_f1s, tfidf_times = tfidf_results
        print("TF-IDF Features:")
        print(f"  Mean Accuracy: {np.mean(tfidf_accs):.4f} ± {np.std(tfidf_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in tfidf_accs]}")
        print(f"  Mean F1 Score: {np.mean(tfidf_f1s):.4f} ± {np.std(tfidf_f1s):.4f}")
        print(f"  Per-fold F1 Scores: {[f'{f1:.4f}' for f1 in tfidf_f1s]}")
        print(f"  Mean Time per Fold: {np.mean(tfidf_times):.2f}s ± {np.std(tfidf_times):.2f}s")
        print(f"  Total Time: {np.sum(tfidf_times):.2f}s")
    
    print("\n" + "-"*80 + "\n")
    
    print(f"\n{'='*80}\n")


def save_results_to_csv(tfidf_results=None, output_file="evaluation_results.csv"):
    """Save results to CSV file."""
    # Create base DataFrame
    results_dict = {
        'Fold': range(1, 6)
    }
    
    # Add TF-IDF results if available
    if tfidf_results is not None:
        tfidf_accs, tfidf_f1s, tfidf_times = tfidf_results
        results_dict['TFIDF_Accuracy'] = tfidf_accs
        results_dict['TFIDF_F1_Score'] = tfidf_f1s
        results_dict['TFIDF_Time'] = tfidf_times
    
    results_df = pd.DataFrame(results_dict)
    
    # Add summary row
    summary_dict = {
        'Fold': ['Mean', 'Std']
    }
    
    if tfidf_results is not None:
        tfidf_accs, tfidf_f1s, tfidf_times = tfidf_results
        summary_dict['TFIDF_Accuracy'] = [np.mean(tfidf_accs), np.std(tfidf_accs)]
        summary_dict['TFIDF_F1_Score'] = [np.mean(tfidf_f1s), np.std(tfidf_f1s)]
        summary_dict['TFIDF_Time'] = [np.mean(tfidf_times), np.std(tfidf_times)]
    
    summary_row = pd.DataFrame(summary_dict)
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    dataset_path = "combined_training_dataset.csv"
    
    # Load and preprocess data (for TF-IDF - using full text, not first sentence)
    X, y = load_and_preprocess_data_full_text(dataset_path)
    
    # Initialize TF-IDF classifier
    tfidf_classifier = TFIDFClassifier()
    
    # Evaluate TF-IDF
    print("\n" + "="*80)
    print("STARTING TF-IDF FEATURES EVALUATION")
    print("="*80)
    tfidf_results = tfidf_classifier.evaluate_cv(X, y, n_splits=5)
    
    # Print summary
    print_results_summary(tfidf_results=tfidf_results)
    
    # Save results
    save_results_to_csv(tfidf_results=tfidf_results)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()


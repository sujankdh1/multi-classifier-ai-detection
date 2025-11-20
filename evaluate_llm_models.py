"""
LLM Text Classification Pipeline
Evaluates Universal Sentence Encoder (USE) and DistilBERT models on binary classification task
using 5-fold cross-validation.
"""

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# NLTK for sentence tokenization
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# TensorFlow and USE imports
import tensorflow as tf
import tensorflow_hub as hub

# Transformers and PyTorch imports
from transformers import AutoTokenizer, AutoModel
import torch

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)


class USEClassifier:
    """Universal Sentence Encoder classifier with 5-fold cross-validation."""
    
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embedding_model = None
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load USE model from TensorFlow Hub."""
        print("Loading Universal Sentence Encoder model...")
        # Configure TensorFlow to avoid mutex issues
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow warnings
        # Set TensorFlow to use single thread to avoid mutex issues
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        try:
            print("Downloading/loading model from TensorFlow Hub (this may take a few minutes)...")
            self.embedding_model = hub.load(self.model_url)
            print("USE model loaded successfully.")
        except Exception as e:
            print(f"Error loading USE model: {e}")
            raise
        
    def get_embeddings(self, texts, batch_size=32):
        """Extract embeddings for texts in batches."""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting USE embeddings"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model(batch).numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("Universal Sentence Encoder - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Extract embeddings
            start_time = time.time()
            X_train_emb = self.get_embeddings(X_train.tolist())
            X_val_emb = self.get_embeddings(X_val.tolist())
            
            # Scale features
            X_train_emb_scaled = self.scaler.fit_transform(X_train_emb)
            X_val_emb_scaled = self.scaler.transform(X_val_emb)
            
            # Train classifier
            self.classifier.fit(X_train_emb_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_emb_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_times


class BERTClassifier:
    """DistilBERT classifier with 5-fold cross-validation."""
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=512, batch_size=32):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load DistilBERT model and tokenizer."""
        print("Loading DistilBERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"DistilBERT model loaded successfully on {self.device}.")
        
    def get_embeddings(self, texts):
        """Extract DistilBERT embeddings using [CLS] token."""
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Extracting DistilBERT embeddings"):
                batch_texts = texts[i:i+self.batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get model output
                outputs = self.model(**encoded)
                
                # Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print("DistilBERT - 5-Fold Cross-Validation")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Extract embeddings
            start_time = time.time()
            X_train_emb = self.get_embeddings(X_train.tolist())
            X_val_emb = self.get_embeddings(X_val.tolist())
            
            # Scale features
            X_train_emb_scaled = self.scaler.fit_transform(X_train_emb)
            X_val_emb_scaled = self.scaler.transform(X_val_emb)
            
            # Train classifier
            self.classifier.fit(X_train_emb_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.classifier.predict(X_val_emb_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_times


def get_first_sentence(text):
    """Extract first sentence from text using NLTK tokenizer."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    if not text:
        return ""
    
    # Skip empty lines and very short lines
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 10:  # Skip very short lines
            sentences = sent_tokenize(line)
            if sentences:
                return sentences[0]
    
    # Fallback: return first 200 characters if no sentence found
    return text[:200].strip()


def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
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
    
    # Extract first sentence from each text
    print("\nExtracting first sentence from each text...")
    X = [get_first_sentence(text) for text in tqdm(X, desc="Processing texts")]
    
    # Filter out empty texts
    valid_indices = [i for i, text in enumerate(X) if text.strip()]
    X = [X[i] for i in valid_indices]
    y = [y[i] for i in valid_indices]
    
    # Limit to 1000 samples for testing
    print(f"\nLimiting to 1000 samples for testing...")
    X = X[:1000]
    y = y[:1000]
    
    print(f"\nFinal dataset size: {len(X)} samples")
    print(f"Sample first sentence (first 100 chars): {X[0][:100]}..." if X else "No samples")
    
    return X, y


def print_results_summary(use_results, bert_results):
    """Print formatted results summary."""
    use_accs, use_times = use_results
    bert_accs, bert_times = bert_results
    
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # USE Results
    print("Universal Sentence Encoder (USE):")
    print(f"  Mean Accuracy: {np.mean(use_accs):.4f} ± {np.std(use_accs):.4f}")
    print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in use_accs]}")
    print(f"  Mean Time per Fold: {np.mean(use_times):.2f}s ± {np.std(use_times):.2f}s")
    print(f"  Total Time: {np.sum(use_times):.2f}s")
    
    print("\n" + "-"*80 + "\n")
    
    # DistilBERT Results
    print("DistilBERT (distilbert-base-uncased):")
    print(f"  Mean Accuracy: {np.mean(bert_accs):.4f} ± {np.std(bert_accs):.4f}")
    print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in bert_accs]}")
    print(f"  Mean Time per Fold: {np.mean(bert_times):.2f}s ± {np.std(bert_times):.2f}s")
    print(f"  Total Time: {np.sum(bert_times):.2f}s")
    
    print("\n" + "-"*80 + "\n")
    
    # Comparison
    print("Comparison:")
    print(f"  Accuracy Difference: {np.mean(bert_accs) - np.mean(use_accs):.4f}")
    print(f"  Speed Ratio (USE/DistilBERT): {np.mean(use_times) / np.mean(bert_times):.2f}x")
    
    print(f"\n{'='*80}\n")


def save_results_to_csv(use_results, bert_results, output_file="evaluation_results.csv"):
    """Save results to CSV file."""
    use_accs, use_times = use_results
    bert_accs, bert_times = bert_results
    
    results_df = pd.DataFrame({
        'Fold': range(1, 6),
        'USE_Accuracy': use_accs,
        'USE_Time': use_times,
        'BERT_Accuracy': bert_accs,
        'BERT_Time': bert_times
    })
    
    # Add summary row
    summary_row = pd.DataFrame({
        'Fold': ['Mean', 'Std'],
        'USE_Accuracy': [np.mean(use_accs), np.std(use_accs)],
        'USE_Time': [np.mean(use_times), np.std(use_times)],
        'BERT_Accuracy': [np.mean(bert_accs), np.std(bert_accs)],
        'BERT_Time': [np.mean(bert_times), np.std(bert_times)]
    })
    
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    dataset_path = "combined_training_dataset.csv"
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(dataset_path)
    
    # Initialize models
    use_classifier = USEClassifier()
    bert_classifier = BERTClassifier()
    
    # Load models
    use_classifier.load_model()
    bert_classifier.load_model()
    
    # Evaluate USE
    print("\n" + "="*80)
    print("STARTING USE EVALUATION")
    print("="*80)
    use_results = use_classifier.evaluate_cv(X, y, n_splits=5)
    
    # Evaluate DistilBERT
    print("\n" + "="*80)
    print("STARTING DISTILBERT EVALUATION")
    print("="*80)
    bert_results = bert_classifier.evaluate_cv(X, y, n_splits=5)
    
    # Print summary
    print_results_summary(use_results, bert_results)
    
    # Save results
    save_results_to_csv(use_results, bert_results)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()


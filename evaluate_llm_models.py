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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# SHAP for model interpretability
import shap
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
# tf.random.set_seed(42)
torch.manual_seed(42)

# Model Configuration
USE_DISTILBERT = True  # Set to False for BERT, True for DistilBERT

if USE_DISTILBERT:
    BERT_MODEL_NAME = "distilbert-base-uncased"
    BERT_BATCH_SIZE = 32
    BERT_MODEL_LABEL = "DistilBERT"
else:
    BERT_MODEL_NAME = "bert-base-uncased"
    BERT_BATCH_SIZE = 16
    BERT_MODEL_LABEL = "BERT"


class USEClassifier:
    """Universal Sentence Encoder classifier with 5-fold cross-validation."""
    
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.embedding_model = None
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load USE model from TensorFlow Hub."""
        # Lazy import TensorFlow - only import when actually needed
        # This avoids random_device errors on systems where TensorFlow has issues
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as e:
            raise ImportError(f"TensorFlow is not available. Cannot load USE model. Error: {e}")
        
        print("Loading Universal Sentence Encoder model...")
        
        # Set TensorFlow to use single thread to avoid mutex issues
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except:
            pass  # Ignore if already configured
        
        # Set random seed (use numpy-based approach if tf.random fails)
        try:
            tf.random.set_seed(42)
        except Exception as e:
            # Fallback: use numpy seed if TensorFlow random fails
            print(f"Warning: Could not set TensorFlow random seed: {e}")
            print("Using numpy random seed instead.")
            np.random.seed(42)
        
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
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
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
    """BERT classifier with 5-fold cross-validation."""
    
    def __init__(self, model_name=None, max_length=512, batch_size=None):
        # Use default from config if not specified
        if model_name is None:
            model_name = BERT_MODEL_NAME
        if batch_size is None:
            batch_size = BERT_BATCH_SIZE
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load BERT/DistilBERT model and tokenizer."""
        print(f"Loading {BERT_MODEL_LABEL} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"{BERT_MODEL_LABEL} model loaded successfully on {self.device}.")
        
    def get_embeddings(self, texts):
        """Extract BERT/DistilBERT embeddings using [CLS] token."""
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Extracting {BERT_MODEL_LABEL} embeddings"):
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
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
        fold_times = []
        
        print(f"\n{'='*60}")
        print(f"{BERT_MODEL_LABEL} - 5-Fold Cross-Validation")
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
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.final_model = None  # Store final trained model for SHAP
        self.final_vectorizer = None  # Store final vectorizer for SHAP
        self.final_scaler = None  # Store final scaler for SHAP
        self.feature_names = None  # Store feature names for SHAP
        
    def evaluate_cv(self, X, y, n_splits=5):
        """Perform 5-fold cross-validation on TF-IDF features."""
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracies = []
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
            
            fold_time = time.time() - start_time
            fold_accuracies.append(accuracy)
            fold_times.append(fold_time)
            
            print(f"  Accuracy: {accuracy:.4f} | Time: {fold_time:.2f}s")
        
        return fold_accuracies, fold_times
    
    def train_final_model(self, X, y):
        """
        Train a final model on all data for SHAP analysis.
        
        Args:
            X: List or array of text data
            y: Array of labels
        
        Returns:
            Tuple of (X_processed, y) where X_processed is the transformed feature matrix
        """
        print("\n" + "="*60)
        print("Training Final Model for SHAP Analysis")
        print("="*60)
        
        # Convert to numpy array if it's a list
        if isinstance(X, list):
            X = np.array(X, dtype=object)
        if isinstance(y, list):
            y = np.array(y)
        
        # Extract TF-IDF features
        print("Extracting TF-IDF features...")
        X_tfidf = self.vectorizer.fit_transform(X.tolist())
        X_tfidf = X_tfidf.toarray()
        
        # Store feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_tfidf)
        
        # Train classifier
        print("Training RandomForest classifier...")
        self.classifier.fit(X_scaled, y)
        
        # Store final components for SHAP
        self.final_model = self.classifier
        self.final_vectorizer = self.vectorizer
        self.final_scaler = self.scaler
        
        print(f"Final model trained on {len(X)} samples with {X_scaled.shape[1]} features.")
        
        return X_scaled, y


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
    
    # Use full dataset (no sample limit)
    # print(f"\nLimiting to 10 samples for testing...")
    # X = X[:10]
    # y = y[:10]
    
    print(f"\nFinal dataset size: {len(X)} samples")
    print(f"Sample first sentence (first 100 chars): {X[0][:100]}..." if X else "No samples")
    
    return X, y


def load_liwc_data(filepath):
    """Load and preprocess LIWC features dataset."""
    print(f"\nLoading LIWC dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"LIWC dataset shape: {df.shape}")
    
    # Metadata columns to exclude
    metadata_cols = ['pageid', 'title', 'content', 'categories', 'is_ai_flagged', 'Segment']
    
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
    
    return X, y


def print_results_summary(use_results=None, bert_results=None, liwc_results=None, tfidf_results=None):
    """Print formatted results summary."""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # USE Results
    # if use_results is not None:
    #     use_accs, use_times = use_results
    #     print("Universal Sentence Encoder (USE):")
    #     print(f"  Mean Accuracy: {np.mean(use_accs):.4f} ± {np.std(use_accs):.4f}")
    #     print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in use_accs]}")
    #     print(f"  Mean Time per Fold: {np.mean(use_times):.2f}s ± {np.std(use_times):.2f}s")
    #     print(f"  Total Time: {np.sum(use_times):.2f}s")
    #     print("\n" + "-"*80 + "\n")
    
    # BERT/DistilBERT Results
    # if bert_results is not None:
    #     bert_accs, bert_times = bert_results
    #     print(f"{BERT_MODEL_LABEL} ({BERT_MODEL_NAME}):")
    #     print(f"  Mean Accuracy: {np.mean(bert_accs):.4f} ± {np.std(bert_accs):.4f}")
    #     print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in bert_accs]}")
    #     print(f"  Mean Time per Fold: {np.mean(bert_times):.2f}s ± {np.std(bert_times):.2f}s")
    #     print(f"  Total Time: {np.sum(bert_times):.2f}s")
    #     print("\n" + "-"*80 + "\n")
    
    # LIWC Results
    # if liwc_results is not None:
    #     liwc_accs, liwc_times = liwc_results
    #     print("LIWC Features:")
    #     print(f"  Mean Accuracy: {np.mean(liwc_accs):.4f} ± {np.std(liwc_accs):.4f}")
    #     print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in liwc_accs]}")
    #     print(f"  Mean Time per Fold: {np.mean(liwc_times):.2f}s ± {np.std(liwc_times):.2f}s")
    #     print(f"  Total Time: {np.sum(liwc_times):.2f}s")
    
    # TF-IDF Results
    if tfidf_results is not None:
        tfidf_accs, tfidf_times = tfidf_results
        print("TF-IDF Features:")
        print(f"  Mean Accuracy: {np.mean(tfidf_accs):.4f} ± {np.std(tfidf_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in tfidf_accs]}")
        print(f"  Mean Time per Fold: {np.mean(tfidf_times):.2f}s ± {np.std(tfidf_times):.2f}s")
        print(f"  Total Time: {np.sum(tfidf_times):.2f}s")
    
    print("\n" + "-"*80 + "\n")
    
    # Comparison (commented out for TF-IDF only branch)
    # print("Comparison:")
    # if use_results is not None and bert_results is not None:
    #     use_accs, use_times = use_results
    #     bert_accs, bert_times = bert_results
    #     print(f"  USE vs {BERT_MODEL_LABEL} Accuracy Difference: {np.mean(bert_accs) - np.mean(use_accs):.4f}")
    #     print(f"  Speed Ratio (USE/{BERT_MODEL_LABEL}): {np.mean(use_times) / np.mean(bert_times):.2f}x")
    # 
    # if liwc_results is not None:
    #     liwc_accs, liwc_times = liwc_results
    #     if use_results is not None:
    #         use_accs, use_times = use_results
    #         print(f"  LIWC vs USE Accuracy Difference: {np.mean(liwc_accs) - np.mean(use_accs):.4f}")
    #         print(f"  Speed Ratio (LIWC/USE): {np.mean(liwc_times) / np.mean(use_times):.2f}x")
    #     if bert_results is not None:
    #         bert_accs, bert_times = bert_results
    #         print(f"  LIWC vs {BERT_MODEL_LABEL} Accuracy Difference: {np.mean(liwc_accs) - np.mean(bert_accs):.4f}")
    #         print(f"  Speed Ratio (LIWC/{BERT_MODEL_LABEL}): {np.mean(liwc_times) / np.mean(bert_times):.2f}x")
    
    print(f"\n{'='*80}\n")


def save_results_to_csv(use_results=None, bert_results=None, liwc_results=None, tfidf_results=None, output_file="evaluation_results.csv"):
    """Save results to CSV file."""
    # Create base DataFrame
    results_dict = {
        'Fold': range(1, 6)
    }
    
    # Add USE results if available
    # if use_results is not None:
    #     use_accs, use_times = use_results
    #     results_dict['USE_Accuracy'] = use_accs
    #     results_dict['USE_Time'] = use_times
    
    # Add BERT results if available
    # if bert_results is not None:
    #     bert_accs, bert_times = bert_results
    #     results_dict['BERT_Accuracy'] = bert_accs
    #     results_dict['BERT_Time'] = bert_times
    
    # Add LIWC results if available
    # if liwc_results is not None:
    #     liwc_accs, liwc_times = liwc_results
    #     results_dict['LIWC_Accuracy'] = liwc_accs
    #     results_dict['LIWC_Time'] = liwc_times
    
    # Add TF-IDF results if available
    if tfidf_results is not None:
        tfidf_accs, tfidf_times = tfidf_results
        results_dict['TFIDF_Accuracy'] = tfidf_accs
        results_dict['TFIDF_Time'] = tfidf_times
    
    results_df = pd.DataFrame(results_dict)
    
    # Add summary row
    summary_dict = {
        'Fold': ['Mean', 'Std']
    }
    
    # if use_results is not None:
    #     use_accs, use_times = use_results
    #     summary_dict['USE_Accuracy'] = [np.mean(use_accs), np.std(use_accs)]
    #     summary_dict['USE_Time'] = [np.mean(use_times), np.std(use_times)]
    # 
    # if bert_results is not None:
    #     bert_accs, bert_times = bert_results
    #     summary_dict['BERT_Accuracy'] = [np.mean(bert_accs), np.std(bert_accs)]
    #     summary_dict['BERT_Time'] = [np.mean(bert_times), np.std(bert_times)]
    # 
    # if liwc_results is not None:
    #     liwc_accs, liwc_times = liwc_results
    #     summary_dict['LIWC_Accuracy'] = [np.mean(liwc_accs), np.std(liwc_accs)]
    #     summary_dict['LIWC_Time'] = [np.mean(liwc_times), np.std(liwc_times)]
    
    if tfidf_results is not None:
        tfidf_accs, tfidf_times = tfidf_results
        summary_dict['TFIDF_Accuracy'] = [np.mean(tfidf_accs), np.std(tfidf_accs)]
        summary_dict['TFIDF_Time'] = [np.mean(tfidf_times), np.std(tfidf_times)]
    
    summary_row = pd.DataFrame(summary_dict)
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def generate_shap_explanations(classifier, X_sample, y_sample=None, max_samples=10, output_dir="shap_results"):
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
    import os
    
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
            X_sample, _, y_sample, _ = train_test_split(  # Fixed: take train set (first), not test set
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
    if actual_samples != max_samples:
        print(f"Warning: Expected {max_samples} samples but got {actual_samples}")
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
            plt.title('Top 30 Features by Mean Absolute SHAP Value')
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
    dataset_path = "combined_training_dataset.csv"
    # liwc_path = "LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv"
    
    # Load and preprocess data (for TF-IDF)
    X, y = load_and_preprocess_data(dataset_path)
    
    # Load LIWC data (commented out for TF-IDF only branch)
    # X_liwc, y_liwc = load_liwc_data(liwc_path)
    
    # Check if TensorFlow can be imported (using subprocess to avoid crashes)
    # print("\nChecking TensorFlow availability for USE model...")
    # tensorflow_available = False
    # try:
    #     # Test TensorFlow import in a subprocess to avoid crashing the main process
    #     result = subprocess.run(
    #         [sys.executable, '-c', 'import tensorflow as tf; import tensorflow_hub as hub; print("OK")'],
    #         capture_output=True,
    #         timeout=10,
    #         text=True
    #     )
    #     if result.returncode == 0:
    #         tensorflow_available = True
    #         print("✓ TensorFlow is available. USE model can be used.")
    #     else:
    #         print("✗ TensorFlow import failed. USE will be skipped.")
    #         print(f"  Error: {result.stderr[:200]}")
    # except subprocess.TimeoutExpired:
    #     print("✗ TensorFlow import timed out. USE will be skipped.")
    # except Exception as e:
    #     print(f"✗ Could not test TensorFlow: {e}. USE will be skipped.")
    
    # Initialize models (commented out for TF-IDF only branch)
    # bert_classifier = BERTClassifier()  # Uses config: BERT_MODEL_NAME and BERT_BATCH_SIZE
    # liwc_classifier = LIWCClassifier()
    tfidf_classifier = TFIDFClassifier()  # TF-IDF classifier
    
    # Try to initialize USE only if TensorFlow is available (commented out)
    # use_classifier = None
    # use_results = None
    # 
    # if tensorflow_available:
    #     print("\nAttempting to initialize USE model...")
    #     try:
    #         use_classifier = USEClassifier()
    #         print("USE classifier created. Loading model...")
    #         use_classifier.load_model()
    #         print("USE model loaded successfully!\n")
    #     except Exception as e:
    #         print(f"\n⚠️  Warning: Could not initialize USE model: {e}")
    #         print("Skipping USE evaluation. Continuing with BERT and LIWC only.\n")
    #         use_classifier = None
    # else:
    #     print("\n⚠️  Skipping USE evaluation (TensorFlow not available).")
    #     print("Continuing with BERT and LIWC only.\n")
    
    # Load models (commented out for TF-IDF only branch)
    # bert_classifier.load_model()
    # LIWC doesn't need model loading - uses features directly
    # TF-IDF doesn't need model loading - uses features directly
    
    # Evaluate USE (if available) - commented out
    # if use_classifier is not None:
    #     print("\n" + "="*80)
    #     print("STARTING USE EVALUATION")
    #     print("="*80)
    #     try:
    #         use_results = use_classifier.evaluate_cv(X, y, n_splits=5)
    #     except Exception as e:
    #         print(f"Error during USE evaluation: {e}")
    #         print("Skipping USE results. Continuing with BERT and LIWC.\n")
    #         use_results = None
    
    # Evaluate BERT/DistilBERT - commented out
    # print("\n" + "="*80)
    # print(f"STARTING {BERT_MODEL_LABEL} EVALUATION")
    # print("="*80)
    # bert_results = bert_classifier.evaluate_cv(X, y, n_splits=5)
    
    # Evaluate LIWC - commented out
    # print("\n" + "="*80)
    # print("STARTING LIWC FEATURES EVALUATION")
    # print("="*80)
    # liwc_results = liwc_classifier.evaluate_cv(X_liwc, y_liwc, n_splits=5)
    
    # Evaluate TF-IDF
    print("\n" + "="*80)
    print("STARTING TF-IDF FEATURES EVALUATION")
    print("="*80)
    # tfidf_results = tfidf_classifier.evaluate_cv(X, y, n_splits=5)
    
    # # Print summary
    # print_results_summary(tfidf_results=tfidf_results)
    
    # # Save results
    # save_results_to_csv(tfidf_results=tfidf_results)
    
    # Generate SHAP explanations
    print("\n" + "="*80)
    print("GENERATING SHAP EXPLANATIONS")
    print("="*80)
    try:
        # Train final model on all data for SHAP
        X_processed, y_processed = tfidf_classifier.train_final_model(X, y)
        
        # Generate SHAP explanations
        shap_values, shap_explainer = generate_shap_explanations(
            tfidf_classifier,
            X_processed,
            y_processed, 
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


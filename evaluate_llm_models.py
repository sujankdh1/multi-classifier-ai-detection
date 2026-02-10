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


def get_first_n_sentences(text, n_sentences=10, max_chars=6000):
    """Extract the first N sentences from text (for USE + BERT)."""
    if not text or pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    # Sentence tokenize; fall back to char-slice if tokenization fails
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = []

    if sentences:
        selected = " ".join(sentences[:n_sentences]).strip()
    else:
        selected = text

    # Keep input reasonably bounded (USE can take long text, but this keeps runtime stable)
    if len(selected) > max_chars:
        selected = selected[:max_chars].rsplit(" ", 1)[0].strip()

    return selected


# Backwards-compatible name used elsewhere in the script
def get_first_sentence(text):
    """Legacy alias: now returns the first ~10 sentences."""
    return get_first_n_sentences(text, n_sentences=10, max_chars=6000)


# Lazy-loaded tokenizer for token-based truncation (USE with 512-token limit)
_truncate_tokenizer = None


def get_truncate_tokenizer():
    """Get or create tokenizer for truncating text to N tokens (same as BERT/DistilBERT)."""
    global _truncate_tokenizer
    if _truncate_tokenizer is None:
        _truncate_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    return _truncate_tokenizer


def truncate_to_max_tokens(text, max_tokens=512):
    """Truncate text to at most max_tokens (WordPiece tokens, same definition as BERT)."""
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    tokenizer = get_truncate_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_tokens, truncation=True)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def load_and_preprocess_data(filepath, text_mode="sentences", n_sentences=10, max_chars=6000, max_tokens=None):
    """Load and preprocess the dataset.

    text_mode:
      - "sentences": use first n_sentences capped by max_chars
      - "full": use the full text; if max_chars is None, no cap (whole text); else capped at max_chars
    max_tokens: if set (e.g. 512), truncate each text to this many tokens (WordPiece, same as BERT) after other preprocessing
    """
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
    
    if text_mode == "full":
        if max_chars is None:
            print("\nUsing full text (no character cap) for USE...")
            X = [str(text).strip() for text in tqdm(X, desc="Processing texts")]
        else:
            print(f"\nUsing full text (capped at {max_chars} chars)...")
            X = [
                str(text)[:max_chars].rsplit(" ", 1)[0].strip() if len(str(text)) > max_chars else str(text).strip()
                for text in tqdm(X, desc="Processing texts")
            ]
    else:
        # Sentence-windowed input (USE-friendly) while BERT/DistilBERT will still respect
        # its tokenizer max_length (512 tokens) via truncation.
        print(f"\nExtracting first {n_sentences} sentences from each text (cap {max_chars} chars)...")
        X = [get_first_n_sentences(text, n_sentences=n_sentences, max_chars=max_chars)
             for text in tqdm(X, desc="Processing texts")]

    if max_tokens is not None:
        print(f"\nTruncating each text to {max_tokens} tokens (WordPiece, for USE)...")
        X = [truncate_to_max_tokens(t, max_tokens=max_tokens) for t in tqdm(X, desc="Truncating to tokens")]

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


def print_results_summary(use_results, bert_results):
    """Print formatted results summary."""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    # USE Results
    if use_results is not None:
        use_accs, use_times = use_results
        print("Universal Sentence Encoder (USE):")
        print(f"  Mean Accuracy: {np.mean(use_accs):.4f} ± {np.std(use_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in use_accs]}")
        print(f"  Mean Time per Fold: {np.mean(use_times):.2f}s ± {np.std(use_times):.2f}s")
        print(f"  Total Time: {np.sum(use_times):.2f}s")
        print("\n" + "-"*80 + "\n")
    
    # BERT/DistilBERT Results
    if bert_results is not None:
        bert_accs, bert_times = bert_results
        print(f"{BERT_MODEL_LABEL} ({BERT_MODEL_NAME}):")
        print(f"  Mean Accuracy: {np.mean(bert_accs):.4f} ± {np.std(bert_accs):.4f}")
        print(f"  Per-fold Accuracies: {[f'{acc:.4f}' for acc in bert_accs]}")
        print(f"  Mean Time per Fold: {np.mean(bert_times):.2f}s ± {np.std(bert_times):.2f}s")
        print(f"  Total Time: {np.sum(bert_times):.2f}s")
        print("\n" + "-"*80 + "\n")
    
    print("\n" + "-"*80 + "\n")
    
    # Comparison
    print("Comparison:")
    if use_results is not None and bert_results is not None:
        use_accs, use_times = use_results
        bert_accs, bert_times = bert_results
        print(f"  USE vs {BERT_MODEL_LABEL} Accuracy Difference: {np.mean(bert_accs) - np.mean(use_accs):.4f}")
        print(f"  Speed Ratio (USE/{BERT_MODEL_LABEL}): {np.mean(use_times) / np.mean(bert_times):.2f}x")
    
    print(f"\n{'='*80}\n")


def save_results_to_csv(use_results, bert_results, output_file="evaluation_results.csv"):
    """Save results to CSV file."""
    # Create base DataFrame
    results_dict = {
        'Fold': range(1, 6)
    }
    
    # Add USE results if available
    if use_results is not None:
        use_accs, use_times = use_results
        results_dict['USE_Accuracy'] = use_accs
        results_dict['USE_Time'] = use_times
    
    # Add BERT results if available
    if bert_results is not None:
        bert_accs, bert_times = bert_results
        results_dict['BERT_Accuracy'] = bert_accs
        results_dict['BERT_Time'] = bert_times
    
    results_df = pd.DataFrame(results_dict)
    
    # Add summary row
    summary_dict = {
        'Fold': ['Mean', 'Std']
    }
    
    if use_results is not None:
        use_accs, use_times = use_results
        summary_dict['USE_Accuracy'] = [np.mean(use_accs), np.std(use_accs)]
        summary_dict['USE_Time'] = [np.mean(use_times), np.std(use_times)]
    
    if bert_results is not None:
        bert_accs, bert_times = bert_results
        summary_dict['BERT_Accuracy'] = [np.mean(bert_accs), np.std(bert_accs)]
        summary_dict['BERT_Time'] = [np.mean(bert_times), np.std(bert_times)]
    
    summary_row = pd.DataFrame(summary_dict)
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """Main evaluation pipeline."""
    dataset_path = "combined_training_dataset.csv"
    RUN_USE = True   # USE only (DistilBERT and LIWC commented out)
    # RUN_DISTILBERT = False  # commented out: DistilBERT not used

    # Load and preprocess data (full text for USE)
    # if RUN_DISTILBERT and not RUN_USE:
    #     X, y = load_and_preprocess_data(dataset_path, text_mode="full", max_chars=20000)
    # else:
    #     X, y = load_and_preprocess_data(dataset_path, text_mode="sentences", n_sentences=10, max_chars=6000)
    X, y = load_and_preprocess_data(dataset_path, text_mode="full", max_chars=None, max_tokens=512)  # USE only, 512 tokens max
    
    # Check if TensorFlow can be imported (using subprocess to avoid crashes)
    # Only needed when RUN_USE is enabled.
    tensorflow_available = False
    if RUN_USE:
        print("\nChecking TensorFlow availability for USE model...")
        try:
            # Test TensorFlow import in a subprocess to avoid crashing the main process
            result = subprocess.run(
                [sys.executable, '-c', 'import tensorflow as tf; import tensorflow_hub as hub; print(\"OK\")'],
                capture_output=True,
                timeout=300,
                text=True
            )
            if result.returncode == 0:
                tensorflow_available = True
                print("✓ TensorFlow is available. USE model can be used.")
            else:
                print("✗ TensorFlow import failed. USE will be skipped.")
                print(f"  Error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("✗ TensorFlow import timed out. USE will be skipped.")
        except Exception as e:
            print(f"✗ Could not test TensorFlow: {e}. USE will be skipped.")
    
    # Initialize models
    bert_classifier = None
    bert_results = None
    # if RUN_DISTILBERT:
    #     bert_classifier = BERTClassifier()  # Uses config: BERT_MODEL_NAME and BERT_BATCH_SIZE

    # Try to initialize USE only if TensorFlow is available
    use_classifier = None
    use_results = None
    
    if RUN_USE and tensorflow_available:
        print("\nAttempting to initialize USE model...")
        try:
            use_classifier = USEClassifier()
            print("USE classifier created. Loading model...")
            use_classifier.load_model()
            print("USE model loaded successfully!\n")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not initialize USE model: {e}")
            print("Skipping USE evaluation.\n")
            use_classifier = None
    elif RUN_USE:
        print("\n⚠️  Skipping USE evaluation (TensorFlow not available).")
        print("USE will be skipped.\n")
    
    # Load models
    # if RUN_DISTILBERT and bert_classifier is not None:
    #     bert_classifier.load_model()

    # Evaluate USE (if available)
    if RUN_USE and use_classifier is not None:
        print("\n" + "="*80)
        print("STARTING USE EVALUATION")
        print("="*80)
        try:
            use_results = use_classifier.evaluate_cv(X, y, n_splits=5)
        except Exception as e:
            print(f"Error during USE evaluation: {e}")
            print("Skipping USE results.\n")
            use_results = None
    
    # Evaluate BERT/DistilBERT (commented out - USE only)
    # if RUN_DISTILBERT and bert_classifier is not None:
    #     print("\n" + "="*80)
    #     print(f"STARTING {BERT_MODEL_LABEL} EVALUATION")
    #     print("="*80)
    #     bert_results = bert_classifier.evaluate_cv(X, y, n_splits=5)

    # Print summary
    print_results_summary(use_results, bert_results)
    
    # Save results
    save_results_to_csv(use_results, bert_results)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()


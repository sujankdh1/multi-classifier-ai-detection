import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# NLTK
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Transformers
import torch
from transformers import AutoTokenizer, AutoModel

np.random.seed(42)
torch.manual_seed(42)

###########################################
# Configuration for MacBook Air
###########################################
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 4            # small batch size for Mac
MAX_LENGTH = 256          # shorter inputs → faster & enough for sentences
CPU_DEVICE = torch.device("cpu")


###########################################
# Helper Functions
###########################################

def get_first_sentence(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    if not text:
        return ""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 10:
            s = sent_tokenize(line)
            if s:
                return s[0]
    return text[:200]


def load_data(filepath):
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['content'])

    X = df["content"].astype(str).tolist()
    y = df["is_ai_flagged"].tolist()

    print("Extracting first sentences...")
    X = [get_first_sentence(x) for x in tqdm(X)]

    # Remove empty
    valid_idx = [i for i, t in enumerate(X) if t.strip()]
    X = [X[i] for i in valid_idx]
    y = [y[i] for i in valid_idx]

    # Limit to 10 samples
    print("\nLimiting to 10 samples for Mac performance testing...")
    X = X[:10]
    y = y[:10]

    print(f"Final size: {len(X)}")
    print("Example:", X[0])
    return X, y


###########################################
# DistilBERT Embedding Extractor
###########################################

class DistilBERTClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(CPU_DEVICE)
        self.model.eval()
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    def get_embeddings(self, texts):
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
                batch = texts[i:i + BATCH_SIZE]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )

                encoded = {k: v.to(CPU_DEVICE) for k, v in encoded.items()}
                outputs = self.model(**encoded)

                cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_vec)

        return np.vstack(embeddings)

    def evaluate_cv(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, times = [], []

        print("\n===== DistilBERT 5-Fold CV =====")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}")

            X_train = [X[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_train = [y[i] for i in train_idx]
            y_val = [y[i] for i in val_idx]

            start = time.time()

            X_train_emb = self.get_embeddings(X_train)
            X_val_emb = self.get_embeddings(X_val)

            X_train_emb = self.scaler.fit_transform(X_train_emb)
            X_val_emb = self.scaler.transform(X_val_emb)

            self.classifier.fit(X_train_emb, y_train)
            preds = self.classifier.predict(X_val_emb)

            acc = accuracy_score(y_val, preds)
            t = time.time() - start

            accs.append(acc)
            times.append(t)

            print(f"Accuracy: {acc:.4f} | Time: {t:.2f}s")

        return accs, times


###########################################
# Main Run
###########################################

def run_pipeline(text_csv):
    # Load and preprocess
    X, y = load_data(text_csv)

    # BERT model
    bert = DistilBERTClassifier()
    bert_results = bert.evaluate_cv(X, y)

    # Summary
    accs, times = bert_results
    print("\n===== SUMMARY =====")
    print(f"Accuracy mean:  {np.mean(accs):.4f}")
    print(f"Accuracy std:   {np.std(accs):.4f}")
    print(f"Times per fold: {times}")
    print(f"Mean time:      {np.mean(times):.2f}s")


run_pipeline("combined_training_dataset.csv")

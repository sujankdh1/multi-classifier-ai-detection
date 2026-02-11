# AI-Generated Text Detection Using Multiple Classification Approaches

This repository contains implementations of various machine learning classifiers for detecting AI-generated text. The project evaluates different feature extraction methods and provides interpretability analysis using SHAP.

## 📁 Project Structure

The project is organized into multiple branches, each implementing a different classification approach:

### Branches

| Branch | Description | Features |
|--------|-------------|----------|
| **`shap-liwc`** | LIWC features with SHAP analysis | Random Forest on linguistic features + SHAP interpretability |
| **`liwc-classifier`** | LIWC features only | Random Forest with LIWC-22 linguistic features |
| **`tfidf-classifier`** | TF-IDF features | TF-IDF vectorization with LinearSVC |
| **`use-distilbert-classifier`** | USE + DistilBERT embeddings | Sentence embeddings with Random Forest |
| **`master`** | Main/stable branch | Base implementation |

## 🌟 Recommended Branch: `shap-liwc`

The **`shap-liwc`** branch provides the most comprehensive analysis with:
- ✅ Cross-validation evaluation (5-fold)
- ✅ Accuracy and F1-score metrics
- ✅ SHAP interpretability visualizations
- ✅ Feature importance analysis
- ✅ Individual prediction explanations

## 📋 Requirements

### Core Dependencies
```
pandas
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Branch-Specific Dependencies

**For `shap-liwc` and `liwc-classifier`:**
```
shap>=0.43.0
matplotlib>=3.7.0
```

**For `tfidf-classifier`:**
```
scikit-learn>=1.3.0
```

**For `use-distilbert-classifier`:**
```
tensorflow>=2.13.0
tensorflow-hub>=0.15.0
transformers>=4.30.0
torch>=2.0.0
tqdm>=4.65.0
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd training
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📊 Data Requirements

⚠️ **Note:** Data files are not included in this repository due to size constraints. You must provide your own datasets.

The code expects the following data files in the project root:

1. **For LIWC classifiers**: `LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv`
   - Must contain LIWC-22 linguistic features
   - Required columns: `is_ai_flagged` (target variable)
   - Metadata columns: `pageid`, `title`, `content`, `categories`, `Segment`, `WC`
   - Generate using [LIWC-22 software](https://www.liwc.app/)

2. **For TF-IDF and embedding-based classifiers**: `combined_training_dataset.csv`
   - Must contain: `content` (text data) and `is_ai_flagged` (labels)
   - Format: CSV with at least two columns - text content and binary labels

## 💻 Usage

### LIWC Classifier with SHAP (Recommended)

```bash
# Switch to shap-liwc branch
git checkout shap-liwc

# Run evaluation
python evaluate_llm_models.py
```

**Output:**
- Console: Cross-validation results (accuracy, F1-score, timing)
- `liwc_evaluation_results.csv`: Fold-by-fold metrics
- `shap_results/`: Directory containing:
  - `shap_summary_bar.png`: Feature importance ranking
  - `shap_summary_beeswarm.png`: Feature impact distribution
  - `shap_feature_importance.png`: Top 30 features visualization
  - `shap_feature_importance.csv`: Feature importance values
  - `shap_waterfall_example_*.png`: Individual prediction explanations
  - `shap_values.csv`: Raw SHAP values

### LIWC Classifier (Without SHAP)

```bash
# Switch to liwc-classifier branch
git checkout liwc-classifier

# Run evaluation
python evaluate_llm_models.py
```

**Output:**
- Console: Cross-validation results
- `liwc_evaluation_results.csv`: Performance metrics

### TF-IDF Classifier

```bash
# Switch to tfidf-classifier branch
git checkout tfidf-classifier

# Run evaluation
python evaluate_llm_models.py
```

**Output:**
- Console: Cross-validation results with progress indicators
- `evaluation_results.csv`: TF-IDF performance metrics

### USE + DistilBERT Classifier

```bash
# Switch to use-distilbert-classifier branch
git checkout use-distilbert-classifier

# Run evaluation
python evaluate_llm_models.py
```

**Output:**
- Console: Embedding extraction progress and CV results
- `evaluation_results.csv`: Model comparison metrics

## 📈 Metrics

All classifiers report:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall (binary classification)
- **Per-fold results**: Performance across all 5 folds
- **Timing information**: Computational efficiency

## 🔍 SHAP Interpretability (shap-liwc branch)

The SHAP analysis provides insights into model predictions:

### 1. Summary Bar Plot
Shows the top 30 most important features ranked by mean absolute SHAP value.

### 2. Beeswarm Plot
Displays feature values and their impact on predictions:
- Red: High feature value
- Blue: Low feature value
- Position: Impact on prediction

### 3. Feature Importance
Quantifies each LIWC feature's contribution to model decisions.

### 4. Waterfall Plots
Explains individual predictions by showing how each feature contributes to the final prediction.

## 📁 Output Files

### Evaluation Results
- `liwc_evaluation_results.csv`: LIWC classifier metrics
- `evaluation_results.csv`: TF-IDF/embedding-based metrics

### SHAP Results (shap-liwc only)
- `shap_results/`: All SHAP visualizations and data
- `shap_values.csv`: Raw SHAP values for each sample and feature
- `shap_feature_importance.csv`: Ranked feature importance

## 🔧 Configuration

### Modify Cross-Validation Splits
Edit the `n_splits` parameter in `main()`:
```python
liwc_results = liwc_classifier.evaluate_cv(X, y, n_splits=5)  # Change 5 to desired number
```

### Adjust SHAP Samples
Edit `max_samples` in the SHAP generation call:
```python
generate_shap_explanations(
    liwc_classifier,
    X_scaled,
    y_processed, 
    max_samples=1000,  # Reduce for faster computation
    output_dir="shap_results"
)
```

### Random Forest Parameters
Modify in the `LIWCClassifier` class:
```python
self.classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
```

## 🐛 Troubleshooting

### Issue: "File not found" error
**Solution**: Ensure your data file paths match the expected names:
- LIWC: `LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv`
- Others: `combined_training_dataset.csv`

### Issue: SHAP analysis is slow
**Solution**: Reduce `max_samples` parameter (default: 1000). Try 100-500 for faster results.

### Issue: Memory error during TF-IDF
**Solution**: The code uses sparse matrices for efficiency. If issues persist, reduce dataset size or use sampling.

### Issue: TensorFlow errors (USE branch)
**Solution**: Ensure TensorFlow is properly installed. The code includes fallback mechanisms if TensorFlow is unavailable.

## 📚 Research Context

This code was developed for research on AI-generated text detection using linguistic features. The LIWC (Linguistic Inquiry and Word Count) features capture psychological and linguistic patterns that may differ between human and AI-generated text.

## 🤝 Contributing

Each branch is self-contained. To add a new classifier:
1. Create a new branch
2. Implement the classifier class with `evaluate_cv()` method
3. Update this README with branch description

## 📄 License

[Add your license here]

## 📧 Contact

[Add your contact information here]

## 🙏 Acknowledgments

- LIWC-22 for linguistic feature extraction
- SHAP library for model interpretability
- Scikit-learn for machine learning implementations

# AI-Generated Text Detection Using Multiple Classification Approaches

This repository contains implementations of various machine learning classifiers for detecting AI-generated text. The project evaluates different feature extraction methods and provides interpretability analysis using SHAP.

## 📁 Project Overview

This repository provides a **unified evaluation framework** that combines four different classification approaches:

### Classifiers Included

| Classifier | Feature Type | Algorithm | Key Features |
|-----------|--------------|-----------|--------------|
| **1. LIWC Features** | Linguistic features | Random Forest | SHAP interpretability analysis |
| **2. TF-IDF** | Text features | LinearSVC | N-gram based features (unigrams + bigrams) |
| **3. Universal Sentence Encoder (USE)** | Semantic embeddings | Random Forest | Pre-trained sentence embeddings |
| **4. BERT/DistilBERT** | Contextual embeddings | Random Forest | Transformer-based embeddings |

### 🌟 Key Features

- ✅ **All-in-one evaluation**: Run all classifiers in a single script
- ✅ **5-fold cross-validation**: Robust performance evaluation
- ✅ **SHAP interpretability**: Understand LIWC model decisions
- ✅ **Flexible configuration**: Enable/disable individual classifiers
- ✅ **Comprehensive metrics**: Accuracy, F1-score, and timing information

## 📋 Requirements

### Core Dependencies
```
pandas
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Classifier-Specific Dependencies

**For LIWC classifier:**
```
shap>=0.43.0
matplotlib>=3.7.0
```

**For TF-IDF classifier:**
```
scikit-learn>=1.3.0
nltk>=3.8.0
```

**For USE and BERT classifiers:**
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

**Note:** All dependencies for all classifiers are included in `requirements.txt`. If you only plan to use specific classifiers, you can install only the required packages (see Classifier-Specific Dependencies above) and disable unused classifiers in the configuration.

## Data Requirements

⚠️ **Note:** Data files are not included in this repository due to size constraints. You must provide your own datasets.

The code expects the following data files in the project root:

1. **For LIWC classifiers**: `LIWC-22 Results - combined_training_dataset - LIWC Analysis.csv`
   - ⚠️ **IMPORTANT:** To run the LIWC classifier, you **MUST** have the **combined LIWC classifier** output file
   - This file is generated using the [LIWC-22 software tool](https://www.liwc.app/)
   - Must contain LIWC-22 linguistic features extracted from your text data
   - Required columns: `is_ai_flagged` (target variable)
   - Metadata columns: `pageid`, `title`, `content`, `categories`, `Segment`, `WC`
   - **How to generate:** 
     1. Install LIWC-22 software from https://www.liwc.app/
     2. Upload your `combined_training_dataset.csv` to the LIWC-22 tool
     3. Run the LIWC analysis to extract linguistic features
     4. Download the output CSV file and rename it to match the expected filename

2. **For TF-IDF and embedding-based classifiers**: `combined_training_dataset.csv`
   - Must contain: `content` (text data) and `is_ai_flagged` (labels)
   - Format: CSV with at least two columns - text content and binary labels

## 💻 Usage

### Basic Usage (Run All Classifiers)

 **Prerequisites:** 
- For LIWC classifier: You must have the **combined LIWC classifier** output file generated from the LIWC-22 tool (see [Data Requirements](#-data-requirements))
- For text-based classifiers: You must have the `combined_training_dataset.csv` file

```bash
# Run all classifiers
python evaluate_llm_models.py
```

This will evaluate all four classifiers sequentially and generate a combined results file.

### Configuration (Enable/Disable Classifiers)

You can selectively run specific classifiers by editing the configuration flags in `evaluate_llm_models.py`:

```python
# Configuration flags - set to True to enable each classifier
RUN_LIWC = True      # LIWC Features Classifier
RUN_TFIDF = True     # TF-IDF Classifier
RUN_USE = True       # Universal Sentence Encoder
RUN_BERT = True      # BERT/DistilBERT Classifier
RUN_SHAP = True      # SHAP analysis for LIWC
```

**Example:** To run only LIWC and TF-IDF:
```python
RUN_LIWC = True
RUN_TFIDF = True
RUN_USE = False
RUN_BERT = False
RUN_SHAP = True
```

### Output Files

**Combined Results:**
- `combined_evaluation_results.csv`: Metrics for all enabled classifiers
- Console output: Detailed fold-by-fold results with timing

**SHAP Analysis (if enabled for LIWC):**
- `shap_results_liwc/`: Directory containing:
  - `shap_summary_bar.png`: Feature importance ranking
  - `shap_summary_beeswarm.png`: Feature impact distribution
  - `shap_feature_importance.png`: Top 30 features visualization
  - `shap_feature_importance.csv`: Feature importance values
  - `shap_waterfall_example_*.png`: Individual prediction explanations (5 examples)
  - `shap_values.csv`: Raw SHAP values for all samples

## 📈 Metrics

All classifiers report:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall (binary classification)
- **Per-fold results**: Performance across all 5 folds
- **Timing information**: Computational efficiency

## 🔍 SHAP Interpretability

The SHAP analysis (for LIWC classifier) provides insights into model predictions:

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
- `combined_evaluation_results.csv`: Comprehensive metrics for all classifiers
  - Per-fold accuracy and F1-scores for each classifier
  - Mean and standard deviation statistics
  - Execution time per fold

### SHAP Results (when enabled)
- `shap_results_liwc/`: Complete SHAP analysis directory
  - Visualizations (PNG files)
  - Feature importance rankings (CSV)
  - Raw SHAP values for further analysis

## 🔧 Configuration

### Enable/Disable Classifiers
Edit configuration flags in `evaluate_llm_models.py` (around line 980):
```python
RUN_LIWC = True      # LIWC Features Classifier
RUN_TFIDF = True     # TF-IDF Classifier
RUN_USE = True       # Universal Sentence Encoder
RUN_BERT = True      # BERT/DistilBERT Classifier
RUN_SHAP = True      # SHAP analysis for LIWC
```

### Choose BERT Model
Switch between BERT and DistilBERT (around line 53):
```python
USE_DISTILBERT = True  # Set to False for BERT, True for DistilBERT
```

### Modify Cross-Validation Splits
Change `n_splits` in the evaluation calls within `main()`:
```python
liwc_results = liwc_classifier.evaluate_cv(X, y, n_splits=5)  # Change to desired number
```

### Adjust SHAP Samples
Modify `max_samples` parameter for faster computation:
```python
generate_shap_explanations(
    liwc_classifier,
    X_scaled,
    y_processed, 
    max_samples=1000,  # Reduce for faster computation (e.g., 100-500)
    output_dir="shap_results_liwc"
)
```

### Model Parameters

**Random Forest (LIWC, USE, BERT):**
```python
RandomForestClassifier(
    n_estimators=100,  # Number of trees
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
```

**TF-IDF Vectorizer:**
```python
TfidfVectorizer(
    max_features=5000,      # Maximum vocabulary size
    ngram_range=(1, 2),     # Unigrams and bigrams
    stop_words='english',
    min_df=2,               # Minimum document frequency
    max_df=0.95            # Maximum document frequency
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

### Issue: TensorFlow errors (USE classifier)
**Solution**: Ensure TensorFlow is properly installed. The code includes automatic checks and will skip USE if TensorFlow is unavailable. Set `RUN_USE = False` to disable it.

## Research Context

This code was developed for research on AI-generated text detection using linguistic features. The LIWC (Linguistic Inquiry and Word Count) features capture psychological and linguistic patterns that may differ between human and AI-generated text.

## Contributing

To add a new classifier to the evaluation pipeline:
1. Create a new classifier class with an `evaluate_cv()` method
2. Add the classifier to the `main()` function in `evaluate_llm_models.py`
3. Add a configuration flag (e.g., `RUN_YOUR_CLASSIFIER`)
4. Update the results summary and saving functions
5. Update this README with the new classifier details


## Acknowledgments

- LIWC-22 for linguistic feature extraction
- SHAP library for model interpretability
- Scikit-learn for machine learning implementations

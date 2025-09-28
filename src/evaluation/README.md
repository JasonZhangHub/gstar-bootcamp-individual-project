# Model Evaluation Module

Comprehensive evaluation framework for benchmarking AI text detection models against existing methods with statistical analysis.

## Components

### 1. `model_evaluation.py`
Core evaluation framework with:
- Performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Statistical significance testing (McNemar, Friedman, Cochran's Q)
- Cross-validation analysis
- Error pattern analysis
- HTML report generation
- Visualization tools

### 2. `baseline_detectors.py`
Implementation of existing AI detection methods:
- **Perplexity-based** (DetectGPT-inspired)
- **Burstiness detector** (variance in sentence/word patterns)
- **GLTR-style** (token probability distribution)
- **N-gram classifier** (character & word n-grams)
- **Zipfian detector** (word frequency distribution)
- **Transformer-based** (optional, using pre-trained models)

### 3. `run_evaluation.py`
Complete evaluation pipeline that:
- Loads test data and models
- Trains baseline detectors
- Runs comprehensive benchmarking
- Performs statistical analysis
- Generates reports and visualizations
- Creates LaTeX tables for papers

## Usage

### Quick Start
```bash
# Basic evaluation
python run_evaluation.py \
    --test_data ../../data/processed/test_data.json \
    --model_path ../../models/ensemble_model.pkl \
    --output_dir ../../results/evaluation
```

### Full Evaluation with Baselines
```bash
python run_evaluation.py \
    --test_data ../../data/processed/test_data.json \
    --train_data ../../data/processed/train_data.json \
    --model_path ../../models/ensemble_model.pkl \
    --output_dir ../../results/evaluation
```

### Skip Baseline Training
```bash
python run_evaluation.py \
    --test_data ../../data/processed/test_data.json \
    --model_path ../../models/ensemble_model.pkl \
    --skip_baselines \
    --output_dir ../../results/evaluation
```

## Output Files

The evaluation generates multiple output files:

1. **HTML Report** (`evaluation_report_*.html`)
   - Comprehensive performance comparison
   - Statistical test results
   - Cross-validation scores
   - Error analysis

2. **CSV Files**
   - `model_comparison_*.csv` - Performance metrics for all models
   - `cv_results_*.csv` - Cross-validation detailed results

3. **JSON Files**
   - `statistical_tests_*.json` - Statistical significance test results
   - `error_analysis_*.json` - Error pattern analysis
   - `evaluation_summary_*.json` - High-level summary

4. **Visualizations** (`evaluation_results_*.png`)
   - Performance bar charts
   - ROC curves
   - Cross-validation box plots
   - Confusion matrices

5. **LaTeX Tables** (`latex_tables_*.tex`)
   - Ready-to-use tables for academic papers

## Statistical Tests

### McNemar's Test
- Pairwise comparison between models
- Tests if performance differences are statistically significant
- p-value < 0.05 indicates significant difference

### Friedman Test
- Compares multiple models simultaneously
- Non-parametric alternative to repeated-measures ANOVA
- Post-hoc Nemenyi test for pairwise comparisons

### Cochran's Q Test
- For binary classification with multiple models
- Tests if success rates differ significantly

## Performance Metrics

### Binary Classification (AI Detection)
- **Accuracy**: Overall correctness
- **Precision**: Of predicted AI texts, how many are actually AI?
- **Recall**: Of actual AI texts, how many were detected?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Cohen's Kappa**: Agreement accounting for chance
- **MCC**: Matthews Correlation Coefficient

### Multi-class Classification (Source Identification)
- **Macro averages**: Unweighted mean across classes
- **Weighted averages**: Weighted by class frequency
- **Per-class metrics**: Individual performance for each AI model

## Baseline Methods Explained

### Perplexity Detector
- Lower perplexity often indicates AI-generated text
- Based on DetectGPT approach
- Measures how "surprised" a language model is by the text

### Burstiness Detector
- Measures variance in sentence lengths and word frequencies
- Human text tends to be more "bursty" (variable)
- AI text often has more consistent patterns

### GLTR Detector
- Analyzes token probability distributions
- Looks at how often top-k probable tokens are used
- AI tends to use high-probability tokens more frequently

### N-gram Detector
- Uses character and word n-gram frequencies
- Random Forest classifier on TF-IDF features
- Captures stylistic patterns

### Zipfian Detector
- Analyzes word frequency distributions
- Tests adherence to Zipf's law
- AI text often shows more regular distributions

## Customization

### Adding New Baselines
```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyDetector(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        # Training logic
        return self

    def predict(self, X):
        # Prediction logic
        return predictions
```

### Custom Metrics
```python
evaluator = ModelEvaluator()
evaluator.custom_metric = lambda y_true, y_pred: your_metric(y_true, y_pred)
```

## Requirements

```bash
pip install scikit-learn scipy matplotlib seaborn plotly pandas numpy tqdm
```

Optional for transformer baseline:
```bash
pip install torch transformers
```

## Example Results Interpretation

### Performance Comparison
```
Model               Accuracy  Precision  Recall   F1
Our Ensemble        0.924     0.918      0.931    0.924
GLTR-Style         0.856     0.842      0.873    0.857
N-gram RF          0.837     0.825      0.851    0.838
Perplexity-Based   0.782     0.771      0.795    0.783
```

### Statistical Significance
```
Our Ensemble vs GLTR-Style: p=0.0012 (Significant)
Our Ensemble vs N-gram RF:  p=0.0008 (Significant)
GLTR vs N-gram RF:          p=0.3421 (Not significant)
```

This indicates our ensemble significantly outperforms baselines.

## Tips for Best Results

1. **Balanced Test Set**: Ensure equal human/AI samples
2. **Diverse Sources**: Include multiple AI models in test data
3. **Sufficient Sample Size**: At least 500 samples for reliable statistics
4. **Cross-validation**: Use 5-10 folds for robust estimates
5. **Multiple Runs**: Consider multiple random seeds for stability

## Citation

If using this evaluation framework, please cite:
```bibtex
@software{ai_detection_eval,
  title = {Comprehensive Evaluation Framework for AI Text Detection},
  year = {2025},
  author = {Your Name}
}
```
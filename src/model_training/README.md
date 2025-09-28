# Model Training Module

Ensemble classifier combining stylometric and semantic features with multi-task learning.

## Script

### 06_train_ensemble_model.py
Trains and evaluates the AI detection model with multi-task learning capabilities.

## Model Architecture

### Base Classifiers
1. **Random Forest**
   - n_estimators: 100
   - max_depth: 20
   - Feature importance analysis

2. **Gradient Boosting**
   - n_estimators: 100
   - learning_rate: 0.1
   - max_depth: 5

3. **Support Vector Machine**
   - RBF kernel
   - Probability calibration

4. **Neural Network (MLP)**
   - Hidden layers: [128, 64, 32]
   - Dropout: 0.5
   - ReLU activation

### Ensemble Strategy
- **Voting Classifier**: Soft voting with weighted contributions
- **Stacking**: Meta-learner combining base predictions
- **Multi-task Learning**: Simultaneous AI detection and source model identification

## Usage

### Basic Training
```bash
python 06_train_ensemble_model.py \
    --features_dir ../../features \
    --output_dir ../../models \
    --test_size 0.2
```

### Advanced Options
```bash
python 06_train_ensemble_model.py \
    --features_dir ../../features \
    --output_dir ../../models \
    --ensemble_type voting \
    --use_multitask \
    --cross_validate \
    --n_folds 5 \
    --feature_selection mutual_info \
    --top_k_features 100
```

## Multi-task Learning

The model can simultaneously:
1. **Binary Classification**: Human vs AI-generated
2. **Source Attribution**: Identify which AI model (GPT-4, Claude, etc.)

Architecture:
```
Input Features → Shared Layers → Task-specific Heads
                                  ├── Binary Classifier
                                  └── Model Identifier
```

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean
- **ROC-AUC**: Discrimination ability
- **Confusion Matrix**: Error analysis
- **Out-of-Distribution**: Performance on unseen models

## Feature Selection

Available methods:
- **Mutual Information**: Non-linear relationships
- **Chi-square**: Statistical significance
- **LASSO**: L1 regularization
- **Random Forest Importance**: Tree-based selection

## Cross-validation

Stratified K-fold with:
- Preserving class distribution
- Topic-based stratification
- Time-based splits (if temporal)

## Output Files

```
models/
├── ensemble_model.pkl        # Main ensemble classifier
├── feature_scaler.pkl        # Feature normalization
├── feature_selector.pkl      # Selected features
├── model_metrics.json        # Performance metrics
├── confusion_matrix.png      # Visualization
├── feature_importance.csv    # Feature rankings
└── multitask_model.pt       # PyTorch multi-task model
```

## Hyperparameter Tuning

Use Grid Search or Bayesian Optimization:
```bash
python 06_train_ensemble_model.py \
    --hyperparameter_search grid \
    --search_params config/hyperparam_grid.json
```

## Benchmarking

Compare against baselines:
- GPTZero
- OpenAI Classifier
- DetectGPT
- GLTR

```bash
python 06_train_ensemble_model.py \
    --benchmark_models gpt_zero openai_detector \
    --save_comparison ../../results/benchmark.csv
```

## Adversarial Testing

Test robustness:
```bash
python 06_train_ensemble_model.py \
    --test_adversarial \
    --adversarial_dir ../../data/adversarial
```
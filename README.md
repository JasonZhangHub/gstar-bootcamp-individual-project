# AI-Generated Media News Detection Project

A hybrid stylometric and semantic fluctuation approach for detecting AI-generated media news using ensemble machine learning and multi-task learning.

## 📁 Project Structure

```
.
├── src/                      # Source code
│   ├── data_collection/      # Data gathering and generation scripts
│   ├── feature_engineering/  # Feature extraction modules
│   ├── model_training/       # Model development and training
│   └── utils/                # Utility functions and helpers
├── data/                     # Raw and processed datasets
├── features/                 # Extracted features cache
├── models/                   # Trained model checkpoints
├── results/                  # Experiment results and outputs
├── experiments/              # Experimental notebooks and scripts
├── tests/                    # Unit and integration tests
├── literature_review/        # Research papers and notes
├── proposal/                 # Project proposal documents
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Pipeline Execution

Execute the pipeline scripts in order:

```bash
# Step 1: Collect human-written news articles
python src/data_collection/01_download_human_datasets.py

# Step 2: Generate AI counterparts
python src/data_collection/02_generate_ai_counterparts.py --models gpt-4 claude

# Step 3: Balance dataset (1:1 ratio)
python src/data_collection/03_balance_dataset.py

# Step 4: Extract features
python src/feature_engineering/04_extract_stylometric_features.py
python src/feature_engineering/05_extract_semantic_features.py

# Step 5: Train ensemble model
python src/model_training/06_train_ensemble_model.py
```

## 📊 Key Components

### Data Collection (`src/data_collection/`)
- **01_download_human_datasets.py**: Downloads human-written news from Reuters, AP, BBC, CNN
- **02_generate_ai_counterparts.py**: Generates AI versions using GPT-4, Claude, Gemini, etc.
- **03_balance_dataset.py**: Creates balanced dataset with equal human/AI samples

### Feature Engineering (`src/feature_engineering/`)
- **04_extract_stylometric_features.py**:
  - Lexical diversity metrics
  - Sentence complexity patterns
  - N-gram frequency distributions
  - Punctuation and formatting patterns

- **05_extract_semantic_features.py**:
  - Topic coherence drift measurement
  - Entity consistency scoring
  - Temporal logic validation
  - Source attribution patterns
  - Factual grounding metrics

### Model Training (`src/model_training/`)
- **06_train_ensemble_model.py**:
  - Ensemble methods (Random Forest, Gradient Boosting, SVM)
  - Multi-task learning (AI detection + source identification)
  - Contrastive learning components
  - Cross-validation and benchmarking

### Testing (`tests/`)
- **test_gpt2_generation.py**: Quick GPT-2 generation test
- **test_huggingface_generation.py**: HuggingFace inference API test

### Utilities (`src/utils/`)
- **load_full_datasets.py**: Dataset loading utilities

## 📈 Expected Outcomes

- Improved detection accuracy on adversarial examples
- Robust performance on out-of-distribution data
- Identification of most discriminative features
- Multi-model source attribution capability

## 🔬 Research Approach

This project implements the methodology described in the [research proposal](proposal/research_proposal_v2.md), combining:

1. **Stylometric Analysis**: Statistical patterns in writing style
2. **Semantic Fluctuation Analysis**: Novel coherence and consistency metrics
3. **Ensemble Learning**: Multiple classifiers for robust predictions
4. **Multi-task Learning**: Simultaneous AI detection and model identification

## 📝 Documentation

- [Research Proposal](proposal/research_proposal_v2.md) - Detailed project methodology
- [Literature Review](literature_review/) - Related work and references
- [Results](results/) - Experiment outputs and analysis

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Experimental Scripts
Place experimental notebooks and scripts in `experiments/` folder.

## 📊 Data Storage

- `data/raw/`: Original datasets
- `data/processed/`: Cleaned and preprocessed data
- `data/generated/`: AI-generated articles
- `features/`: Extracted feature matrices
- `models/`: Trained model checkpoints
- `results/`: Evaluation metrics and plots

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Document your changes
4. Keep scripts modular and reusable

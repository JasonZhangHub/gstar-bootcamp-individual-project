# Data Processing Module

Scripts for preparing and processing data for model training and evaluation.

## Main Script

### `prepare_evaluation_data.py`
Comprehensive data preparation pipeline that:
- Loads human-written articles from JSON files
- Loads AI-generated articles from multiple sources
- Balances the dataset (equal human/AI samples)
- Extracts features (basic, stylometric, semantic)
- Splits data into train/validation/test sets
- Saves processed data in multiple formats

## Usage

### Quick Start (Basic Processing)
```bash
# Process data with basic features only
python prepare_evaluation_data.py \
    --human_data ../../data/datasets/combined_human_news_full.json \
    --ai_data ../../data/ai_generated \
    --output_dir ../../data/processed \
    --max_samples 500
```

### Full Feature Extraction
```bash
# Include stylometric features (recommended)
python prepare_evaluation_data.py \
    --human_data ../../data/datasets/combined_human_news_full.json \
    --ai_data ../../data/ai_generated \
    --output_dir ../../data/processed \
    --max_samples 1000

# Include both stylometric and semantic features (slower)
python prepare_evaluation_data.py \
    --human_data ../../data/datasets/combined_human_news_full.json \
    --ai_data ../../data/ai_generated \
    --output_dir ../../data/processed \
    --max_samples 500 \
    --semantic
```

### Custom Data Split
```bash
# Adjust train/val/test proportions
python prepare_evaluation_data.py \
    --human_data ../../data/datasets/combined_human_news_full.json \
    --ai_data ../../data/ai_generated \
    --output_dir ../../data/processed \
    --test_size 0.3 \
    --val_size 0.15
```

## Input Data Format

### Human Data (JSON)
```json
[
  {
    "text": "Article content...",
    "id": "article_123",
    "topic": "politics",
    "source": "reuters"
  }
]
```

### AI Data (JSON)
```json
[
  {
    "generated_text": "AI-generated content...",
    "model": "gpt-4",
    "original_id": "human_article_123",
    "topic": "politics"
  }
]
```

## Output Files

The script generates the following files in `data/processed/`:

### JSON Files (Full Data)
- `train_data.json` - Training set with features
- `val_data.json` - Validation set with features
- `test_data.json` - Test set with features

### CSV Files (For Inspection)
- `train_data.csv` - Training set summary
- `val_data.csv` - Validation set summary
- `test_data.csv` - Test set summary

### Metadata
- `data_metadata.json` - Processing information and statistics

## Output Data Format

Each processed article contains:
```json
{
  "id": "unique_id",
  "text": "Full article text...",
  "source": "human" or "ai",
  "label": 0 or 1,  // 0=human, 1=AI
  "model": "gpt-4",  // For AI articles
  "topic": "category",
  "features": {
    "word_count": 250,
    "sentence_count": 12,
    "avg_word_length": 4.5,
    "lexical_diversity": 0.65,
    // ... more features
  }
}
```

## Features Extracted

### Basic Features (Always Included)
- Word count
- Sentence count
- Average word/sentence length
- Lexical diversity
- Punctuation usage
- Capitalization ratio

### Stylometric Features (Default)
- Type-token ratio
- Hapax legomena
- N-gram frequencies
- Syntactic complexity
- POS tag distributions

### Semantic Features (Optional)
- Topic coherence
- Entity consistency
- Temporal patterns
- Source attribution
- Factual grounding

## Complete Workflow

### 1. Prepare Data
```bash
# Run the preparation script
./prepare_test_data.sh
```

### 2. Train Model
```bash
python src/model_training/06_train_ensemble_model.py \
    --train_data data/processed/train_data.json \
    --val_data data/processed/val_data.json
```

### 3. Run Evaluation
```bash
python src/evaluation/run_evaluation.py \
    --test_data data/processed/test_data.json \
    --model_path models/ensemble_model.pkl
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--human_data` | Required | Path to human articles JSON |
| `--ai_data` | Required | Directory with AI articles |
| `--output_dir` | `../../data/processed` | Output directory |
| `--max_samples` | None (all) | Max samples per class |
| `--no_features` | False | Skip feature extraction |
| `--semantic` | False | Include semantic features |
| `--test_size` | 0.2 | Test set proportion |
| `--val_size` | 0.1 | Validation set proportion |

## Tips

1. **Sample Size**: Start with 500-1000 samples for quick testing
2. **Features**: Basic + stylometric is usually sufficient
3. **Balance**: Always maintain 1:1 human:AI ratio
4. **Validation**: Use validation set for hyperparameter tuning
5. **Test Set**: Keep test set untouched until final evaluation

## Troubleshooting

### Memory Issues
- Reduce `--max_samples`
- Skip semantic features with `--no_features`
- Process in batches

### Missing Dependencies
```bash
pip install pandas numpy tqdm scikit-learn
pip install spacy nltk
python -m spacy download en_core_web_sm
```

### Data Format Issues
- Ensure JSON files are properly formatted
- Check that text fields are not empty
- Verify article IDs are unique
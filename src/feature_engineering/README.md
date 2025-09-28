# Feature Engineering Module

Scripts for extracting stylometric and semantic features from text for AI detection.

## Scripts

### 04_extract_stylometric_features.py
Extracts linguistic and stylistic features from text.

**Features Extracted:**
- **Lexical Diversity**:
  - Type-Token Ratio (TTR)
  - Hapax legomena (unique words)
  - Yule's K statistic

- **Sentence Complexity**:
  - Average sentence length
  - Parse tree depth
  - Dependency distances

- **N-gram Patterns**:
  - Character n-grams (2-4)
  - Word n-grams (1-3)
  - POS tag sequences

- **Punctuation Patterns**:
  - Punctuation frequency
  - Comma usage patterns
  - Quote and parenthesis usage

**Usage:**
```bash
python 04_extract_stylometric_features.py \
    --input_dir ../../data/processed \
    --output_dir ../../features/stylometric
```

### 05_extract_semantic_features.py
Analyzes semantic coherence and consistency patterns (Novel contribution).

**Features Extracted:**
- **Topic Coherence Drift**:
  - Paragraph-to-paragraph similarity
  - Topic transition smoothness
  - Semantic distance metrics

- **Entity Consistency Score**:
  - Named entity tracking
  - Coreference resolution patterns
  - Entity attribute consistency

- **Temporal Logic Patterns**:
  - Event sequence validation
  - Temporal expression consistency
  - Timeline coherence

- **Source Attribution**:
  - Citation frequency and specificity
  - Quote attribution patterns
  - Source diversity metrics

- **Factual Grounding**:
  - Verifiable claim density
  - Specificity vs vagueness ratio
  - Statistical claim patterns

**Usage:**
```bash
python 05_extract_semantic_features.py \
    --input_dir ../../data/processed \
    --output_dir ../../features/semantic \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

## Output Format

Features are saved as CSV files with article IDs:
```csv
article_id, feature_1, feature_2, ..., feature_n, label
art_001, 0.65, 12.3, ..., 0.89, human
art_002, 0.72, 14.1, ..., 0.76, ai
```

## Feature Combination

To combine all features:
```python
import pandas as pd

stylometric = pd.read_csv('../../features/stylometric/features.csv')
semantic = pd.read_csv('../../features/semantic/features.csv')

# Merge on article_id
combined = pd.merge(stylometric, semantic, on='article_id')
combined.to_csv('../../features/combined_features.csv', index=False)
```

## Dependencies

- spaCy (with en_core_web_sm model)
- NLTK
- sentence-transformers
- scikit-learn
- numpy, pandas

## Performance Notes

- Stylometric extraction: ~100 articles/minute
- Semantic extraction: ~20 articles/minute (due to embeddings)
- Use `--batch_size` flag for memory management
- Enable GPU with `--use_gpu` for faster semantic processing
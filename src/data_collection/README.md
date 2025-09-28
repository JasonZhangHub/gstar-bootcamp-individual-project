# Data Collection Module

Scripts for gathering human-written news articles and generating AI counterparts.

## Scripts

### 01_download_human_datasets.py
Downloads human-written news articles from various sources.

**Sources:**
- Reuters
- Associated Press (AP)
- BBC News
- CNN
- AG News Dataset

**Usage:**
```bash
python 01_download_human_datasets.py --output_dir ../../data/raw/human
```

### 02_generate_ai_counterparts.py
Generates AI versions of human articles using multiple language models.

**Supported Models:**
- OpenAI: GPT-3.5, GPT-4
- Anthropic: Claude
- Google: Gemini
- Open Source: Llama, Mistral, GPT-2

**Usage:**
```bash
python 02_generate_ai_counterparts.py \
    --input_dir ../../data/raw/human \
    --output_dir ../../data/generated \
    --models gpt-4 claude gemini
```

### 03_balance_dataset.py
Creates a balanced dataset with equal ratios of human and AI-generated articles.

**Features:**
- 1:1 human to AI ratio
- Stratified sampling by topic
- Train/test/validation splits

**Usage:**
```bash
python 03_balance_dataset.py \
    --human_dir ../../data/raw/human \
    --ai_dir ../../data/generated \
    --output_dir ../../data/processed
```

## Data Format

All scripts output data in JSON format:
```json
{
  "id": "unique_article_id",
  "text": "Article content...",
  "source": "human" or "ai",
  "model": "gpt-4" (if AI-generated),
  "topic": "politics/technology/sports/etc",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Configuration

Set API keys as environment variables:
```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export HUGGINGFACE_TOKEN="your_token"
```
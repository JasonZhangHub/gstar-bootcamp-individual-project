#!/usr/bin/env python3
"""
Download Open-Source News Datasets for AI-Generated Text Detection
Alternative to web scraping - uses curated, pre-collected datasets
"""

import os
import json
import pandas as pd
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List

class DatasetDownloader:
    """Downloads and processes open-source news datasets"""

    def __init__(self, data_dir: str = "../../data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, filename: str) -> Path:
        """Download a file from URL"""
        filepath = self.data_dir / filename

        if filepath.exists():
            print(f"File {filename} already exists, skipping download")
            return filepath

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"Progress: {percent:.1f}%", end='\r')

        print(f"\n✓ Downloaded {filename}")
        return filepath

    def extract_archive(self, filepath: Path) -> Path:
        """Extract zip or tar archives"""
        extract_dir = filepath.parent / filepath.stem

        if extract_dir.exists():
            print(f"Archive already extracted to {extract_dir}")
            return extract_dir

        print(f"Extracting {filepath.name}...")

        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(filepath, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)

        print(f"✓ Extracted to {extract_dir}")
        return extract_dir

    def get_dataset_info(self) -> Dict[str, Dict]:
        """Information about available datasets"""
        return {
            "all-the-news": {
                "name": "All The News Dataset",
                "description": "143,000 articles from 15 American publications (CNN, Reuters, Guardian, etc.)",
                "source": "Components available on Kaggle",
                "format": "CSV",
                "size": "~250MB",
                "url": "https://components.one/datasets/all-the-news-2-news-articles-dataset/",
                "kaggle": "snapcrack/all-the-news",
                "fields": ["title", "author", "date", "article", "publication"]
            },
            "cc-news": {
                "name": "CC-News (Common Crawl News)",
                "description": "60M+ news articles from CommonCrawl, multilingual",
                "source": "Hugging Face Datasets",
                "format": "JSON/Parquet",
                "size": "Large (streaming recommended)",
                "huggingface": "cc_news",
                "sample_code": "from datasets import load_dataset\nds = load_dataset('cc_news', split='train', streaming=True)"
            },
            "newsroom": {
                "name": "NEWSROOM Dataset",
                "description": "1.3M articles with summaries from 38 publishers",
                "source": "Cornell/Google Research",
                "format": "JSON",
                "size": "~3GB",
                "url": "https://lil.nlp.cornell.edu/newsroom/",
                "paper": "https://arxiv.org/abs/1804.11283",
                "fields": ["title", "text", "summary", "url", "date"]
            },
            "realnews": {
                "name": "RealNews (from Grover paper)",
                "description": "5000 news articles from GPT-2 era, used in neural fake news research",
                "source": "Allen Institute for AI",
                "format": "JSON",
                "size": "~50MB",
                "github": "rowanz/grover",
                "paper": "https://arxiv.org/abs/1905.12616"
            },
            "ag-news": {
                "name": "AG News Classification Dataset",
                "description": "120K news articles in 4 categories (World, Sports, Business, Tech)",
                "source": "Hugging Face / PyTorch",
                "format": "CSV",
                "size": "~30MB",
                "huggingface": "ag_news",
                "sample_code": "from datasets import load_dataset\nds = load_dataset('ag_news')"
            },
            "bbc-news": {
                "name": "BBC News Articles",
                "description": "2225 BBC articles from 2004-2005 in 5 categories",
                "source": "D. Greene and P. Cunningham",
                "format": "Text files",
                "size": "~5MB",
                "url": "http://mlg.ucd.ie/datasets/bbc.html",
                "categories": ["business", "entertainment", "politics", "sport", "tech"]
            },
            "multi-news": {
                "name": "Multi-News",
                "description": "56K article clusters from 1500+ news sources",
                "source": "Yale LILY Lab",
                "format": "JSON",
                "size": "~250MB",
                "github": "Alex-Fabbri/Multi-News",
                "paper": "https://arxiv.org/abs/1906.01749"
            },
            "xsum": {
                "name": "XSum (BBC Summarization)",
                "description": "226K BBC articles with single sentence summaries",
                "source": "Edinburgh NLP",
                "format": "JSON",
                "size": "~250MB",
                "huggingface": "xsum",
                "sample_code": "from datasets import load_dataset\nds = load_dataset('xsum')"
            }
        }

    def download_huggingface_sample(self, dataset_name: str, sample_size: int = 1000):
        """Download sample from Hugging Face dataset"""
        print(f"\nDownloading {sample_size} samples from {dataset_name}...")

        try:
            from datasets import load_dataset

            # Load dataset
            if dataset_name == "cc_news":
                ds = load_dataset(dataset_name, split='train', streaming=True)
                samples = []
                for i, item in enumerate(ds):
                    if i >= sample_size:
                        break
                    samples.append(item)
            else:
                ds = load_dataset(dataset_name, split='train')
                samples = ds.select(range(min(sample_size, len(ds))))
                samples = [dict(sample) for sample in samples]

            # Save to JSON
            output_file = self.data_dir / f"{dataset_name}_sample.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)

            print(f"✓ Saved {len(samples)} samples to {output_file}")
            return output_file

        except ImportError:
            print("Please install datasets library: pip install datasets")
            return None
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return None

    def print_dataset_info(self):
        """Print information about available datasets"""
        datasets = self.get_dataset_info()

        print("\n" + "=" * 80)
        print("AVAILABLE OPEN-SOURCE NEWS DATASETS")
        print("=" * 80)

        for key, info in datasets.items():
            print(f"\n{key.upper()}: {info['name']}")
            print("-" * 40)
            print(f"Description: {info['description']}")
            print(f"Source: {info['source']}")
            print(f"Format: {info['format']}")
            print(f"Size: {info['size']}")

            if 'url' in info:
                print(f"URL: {info['url']}")
            if 'huggingface' in info:
                print(f"Hugging Face: {info['huggingface']}")
            if 'kaggle' in info:
                print(f"Kaggle: {info['kaggle']}")
            if 'github' in info:
                print(f"GitHub: {info['github']}")

def create_download_script():
    """Create a simple bash script for downloading datasets"""
    script_content = """#!/bin/bash
# Download script for news datasets

echo "Setting up dataset download..."

# Create data directory
mkdir -p ../../data/datasets

# Install required Python packages
pip install datasets pandas requests

echo "
========================================
RECOMMENDED DATASETS FOR YOUR PROJECT
========================================

1. AG News (Quick Start - 30MB):
   python -c \"from datasets import load_dataset; ds = load_dataset('ag_news'); ds.save_to_disk('../../data/datasets/ag_news')\"

2. XSum BBC Articles (Medium - 250MB):
   python -c \"from datasets import load_dataset; ds = load_dataset('xsum'); ds.save_to_disk('../../data/datasets/xsum')\"

3. All The News (Large - via Kaggle):
   kaggle datasets download -d snapcrack/all-the-news

4. RealNews (From Grover paper):
   wget https://storage.googleapis.com/grover-models/realnews.tar.gz
   tar -xzf realnews.tar.gz -C ../../data/datasets/

Choose based on your needs:
- Testing: Use AG News or BBC News (small)
- Development: Use XSum or Multi-News (medium)
- Production: Use All The News or CC-News (large)
"

echo "Run the commands above to download your chosen dataset."
"""

    script_path = Path("../../scripts/download_datasets.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"Created download script: {script_path}")

def main():
    """Main execution"""
    downloader = DatasetDownloader("../../data/datasets")

    # Print available datasets
    downloader.print_dataset_info()

    print("\n" + "=" * 80)
    print("QUICK START SAMPLES")
    print("=" * 80)
    print("\nDownloading small samples from Hugging Face datasets...")

    # Download small samples
    datasets_to_sample = ["ag_news", "xsum"]

    for dataset in datasets_to_sample:
        downloader.download_huggingface_sample(dataset, sample_size=100)

    # Create download script1
    create_download_script()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. For quick testing: Use the downloaded AG News samples
2. For full datasets: Run ./scripts/download_datasets.sh
3. For web scraping: Fix the news_scraper.py Reuters issue
4. Recommended: Start with AG News or XSum for initial development
    """)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Load FULL news datasets for AI-generated text detection
Downloads complete datasets, not just samples
Requires: pip install datasets pandas tqdm
"""

import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import time

def load_and_save_full_datasets(output_dir="../../data/datasets", max_articles_per_dataset=None):
    """
    Load full news datasets and save them locally

    Args:
        output_dir: Directory to save datasets
        max_articles_per_dataset: Optional limit for testing (None = full dataset)
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING FULL NEWS DATASETS")
    print("=" * 60)
    if max_articles_per_dataset:
        print(f"NOTE: Limiting to {max_articles_per_dataset} articles per dataset for testing")
    else:
        print("NOTE: Downloading FULL datasets (this may take a while)")
    print("=" * 60)

    # 1. Load AG News (Full: 120k train + 7.6k test)
    print("\n1. Loading AG News dataset (FULL: 127,600 articles)...")
    try:
        start_time = time.time()
        ag_news = load_dataset('ag_news')

        # Process training data
        train_samples = []
        train_data = ag_news['train']
        limit = min(len(train_data), max_articles_per_dataset) if max_articles_per_dataset else len(train_data)

        print(f"   Processing {limit} training samples...")
        for i in tqdm(range(limit), desc="   AG News Train"):
            item = train_data[i]
            train_samples.append({
                'text': item['text'],
                'label': item['label'],
                'label_name': ['World', 'Sports', 'Business', 'Technology'][item['label']],
                'source': 'ag_news',
                'dataset': 'train'
            })

        # Process test data
        test_samples = []
        test_data = ag_news['test']
        test_limit = min(len(test_data), max_articles_per_dataset // 10) if max_articles_per_dataset else len(test_data)

        print(f"   Processing {test_limit} test samples...")
        for i in tqdm(range(test_limit), desc="   AG News Test"):
            item = test_data[i]
            test_samples.append({
                'text': item['text'],
                'label': item['label'],
                'label_name': ['World', 'Sports', 'Business', 'Technology'][item['label']],
                'source': 'ag_news',
                'dataset': 'test'
            })

        # Save to JSON
        ag_news_path = output_path / 'ag_news_full.json'
        print("   Saving to JSON...")
        with open(ag_news_path, 'w', encoding='utf-8') as f:
            json.dump({
                'train': train_samples,
                'test': test_samples,
                'info': {
                    'train_size': len(train_samples),
                    'test_size': len(test_samples),
                    'categories': ['World', 'Sports', 'Business', 'Technology'],
                    'total_size': len(train_samples) + len(test_samples)
                }
            }, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"✓ Saved AG News: {len(train_samples)} train, {len(test_samples)} test samples")
        print(f"  File: {ag_news_path}")
        print(f"  Time: {elapsed:.1f} seconds")

    except Exception as e:
        print(f"✗ Error loading AG News: {e}")

    # 2. Load CNN/DailyMail (Full: 287k train + 13k val + 11k test)
    print("\n2. Loading CNN/DailyMail dataset (FULL: 311,971 articles)...")
    try:
        start_time = time.time()
        cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0')

        cnn_samples = []

        # Process all splits
        for split_name in ['train', 'validation', 'test']:
            split_data = cnn_dailymail[split_name]
            limit = min(len(split_data), max_articles_per_dataset // 3) if max_articles_per_dataset else len(split_data)

            print(f"   Processing {limit} {split_name} samples...")
            for i in tqdm(range(limit), desc=f"   CNN/DM {split_name}"):
                item = split_data[i]
                cnn_samples.append({
                    'article': item['article'],
                    'highlights': item['highlights'],
                    'id': item['id'],
                    'source': 'cnn_dailymail',
                    'dataset': split_name
                })

        # Save to JSON
        cnn_path = output_path / 'cnn_dailymail_full.json'
        print("   Saving to JSON...")
        with open(cnn_path, 'w', encoding='utf-8') as f:
            json.dump({
                'articles': cnn_samples,
                'info': {
                    'size': len(cnn_samples),
                    'sources': ['CNN', 'Daily Mail'],
                    'features': ['article', 'highlights'],
                    'splits': {split: sum(1 for s in cnn_samples if s['dataset'] == split)
                              for split in ['train', 'validation', 'test']}
                }
            }, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"✓ Saved CNN/DailyMail: {len(cnn_samples)} articles")
        print(f"  File: {cnn_path}")
        print(f"  Time: {elapsed:.1f} seconds")

    except Exception as e:
        print(f"✗ Error loading CNN/DailyMail: {e}")

    # 3. Load BBC News Summary (XSum alternative - Full: 2225 articles)
    print("\n3. Loading BBC News dataset (2225 articles)...")
    try:
        start_time = time.time()

        # Since XSum has issues, let's use the BBC news dataset from Kaggle
        # For now, we'll create a placeholder
        bbc_info = {
            'name': 'BBC News',
            'description': 'BBC News articles from 2004-2005',
            'categories': ['business', 'entertainment', 'politics', 'sport', 'tech'],
            'size': 2225,
            'download_info': 'Download from: http://mlg.ucd.ie/datasets/bbc.html',
            'alternative': 'Or use Kaggle: kaggle datasets download -d pariza/bbc-news-summary'
        }

        bbc_path = output_path / 'bbc_news_info.json'
        with open(bbc_path, 'w', encoding='utf-8') as f:
            json.dump(bbc_info, f, indent=2)

        print(f"✓ BBC News info saved (manual download required)")
        print(f"  File: {bbc_path}")
        print(f"  Download from: {bbc_info['download_info']}")

    except Exception as e:
        print(f"✗ Error with BBC News: {e}")

    # 4. Load Multi-News dataset (Full: 44k train + 5.6k val + 5.6k test)
    print("\n4. Loading Multi-News dataset (FULL: 55,972 article clusters)...")
    try:
        start_time = time.time()
        multi_news = load_dataset('multi_news')

        multi_samples = []

        # Process all splits
        for split_name in ['train', 'validation', 'test']:
            split_data = multi_news[split_name]
            limit = min(len(split_data), max_articles_per_dataset // 3) if max_articles_per_dataset else len(split_data)

            print(f"   Processing {limit} {split_name} samples...")
            for i in tqdm(range(limit), desc=f"   Multi-News {split_name}"):
                item = split_data[i]
                multi_samples.append({
                    'document': item['document'],
                    'summary': item['summary'],
                    'source': 'multi_news',
                    'dataset': split_name
                })

        # Save to JSON
        multi_path = output_path / 'multi_news_full.json'
        print("   Saving to JSON...")
        with open(multi_path, 'w', encoding='utf-8') as f:
            json.dump({
                'articles': multi_samples,
                'info': {
                    'size': len(multi_samples),
                    'description': 'Multi-document news summarization dataset',
                    'features': ['document', 'summary'],
                    'splits': {split: sum(1 for s in multi_samples if s['dataset'] == split)
                              for split in ['train', 'validation', 'test']}
                }
            }, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"✓ Saved Multi-News: {len(multi_samples)} article clusters")
        print(f"  File: {multi_path}")
        print(f"  Time: {elapsed:.1f} seconds")

    except Exception as e:
        print(f"✗ Error loading Multi-News: {e}")

    # 5. Create combined dataset for training
    print("\n5. Creating combined dataset from all sources...")
    try:
        start_time = time.time()
        combined_data = []

        # Add AG News samples
        if (output_path / 'ag_news_full.json').exists():
            with open(output_path / 'ag_news_full.json', 'r') as f:
                ag_data = json.load(f)
                print(f"   Adding {len(ag_data['train'])} AG News articles...")
                for item in tqdm(ag_data['train'], desc="   AG News"):
                    combined_data.append({
                        'text': item['text'],
                        'source': 'ag_news',
                        'category': item['label_name'],
                        'is_human': True,
                        'length': len(item['text'])
                    })

        # Add CNN/DailyMail samples
        if (output_path / 'cnn_dailymail_full.json').exists():
            with open(output_path / 'cnn_dailymail_full.json', 'r') as f:
                cnn_data = json.load(f)
                articles = [a for a in cnn_data['articles'] if a['dataset'] == 'train']
                print(f"   Adding {len(articles)} CNN/DailyMail articles...")
                for item in tqdm(articles, desc="   CNN/DM"):
                    combined_data.append({
                        'text': item['article'],
                        'source': 'cnn_dailymail',
                        'category': 'news',
                        'is_human': True,
                        'length': len(item['article'])
                    })

        # Add Multi-News samples
        if (output_path / 'multi_news_full.json').exists():
            with open(output_path / 'multi_news_full.json', 'r') as f:
                multi_data = json.load(f)
                articles = [a for a in multi_data['articles'] if a['dataset'] == 'train'][:10000]  # Limit for manageable size
                print(f"   Adding {len(articles)} Multi-News articles...")
                for item in tqdm(articles, desc="   Multi-News"):
                    combined_data.append({
                        'text': item['document'],
                        'source': 'multi_news',
                        'category': 'news',
                        'is_human': True,
                        'length': len(item['document'])
                    })

        # Save combined dataset
        combined_path = output_path / 'combined_human_news_full.json'
        print("   Saving combined JSON...")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        # Also save as CSV for easier inspection
        print("   Saving combined CSV...")
        df = pd.DataFrame(combined_data)
        df.to_csv(output_path / 'combined_human_news_full.csv', index=False)

        # Save a smaller sample CSV for quick inspection
        df_sample = df.sample(min(1000, len(df)))
        df_sample.to_csv(output_path / 'combined_human_news_sample.csv', index=False)

        elapsed = time.time() - start_time
        print(f"✓ Created combined dataset: {len(combined_data)} articles")
        print(f"  Full JSON: {combined_path}")
        print(f"  Full CSV: {output_path / 'combined_human_news_full.csv'}")
        print(f"  Sample CSV: {output_path / 'combined_human_news_sample.csv'}")
        print(f"  Time: {elapsed:.1f} seconds")

        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Total articles: {len(combined_data):,}")
        print(f"  Sources: {df['source'].value_counts().to_dict()}")
        print(f"  Average text length: {df['length'].mean():.0f} characters")
        print(f"  Min length: {df['length'].min()} characters")
        print(f"  Max length: {df['length'].max():,} characters")

    except Exception as e:
        print(f"✗ Error creating combined dataset: {e}")

    print("\n" + "=" * 60)
    print("DATASET LOADING COMPLETE")
    print("=" * 60)
    print("\nDatasets are ready for AI-generated text detection research!")
    print("\nNext steps:")
    print("1. Generate AI counterparts using GPT-4/Claude/Gemini")
    print("2. Extract stylometric and semantic fluctuation features")
    print("3. Train your hybrid detection model")
    print("\nTo download even larger datasets, consider:")
    print("- All The News 2.0: kaggle datasets download -d snapcrack/all-the-news")
    print("- CC-News: Use streaming mode for 60M+ articles")
    print("- RealNews: wget https://storage.googleapis.com/grover-models/realnews.tar.gz")

def main():
    """Main execution with options"""
    import argparse

    parser = argparse.ArgumentParser(description='Download full news datasets for AI detection')
    parser.add_argument('--output-dir', default='../../data/datasets',
                       help='Output directory for datasets')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: download only 1000 articles per dataset')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of articles per dataset')

    args = parser.parse_args()

    if args.test_mode:
        print("Running in TEST MODE (limited data)")
        load_and_save_full_datasets(args.output_dir, max_articles_per_dataset=1000)
    else:
        load_and_save_full_datasets(args.output_dir, max_articles_per_dataset=args.limit)

if __name__ == "__main__":
    main()
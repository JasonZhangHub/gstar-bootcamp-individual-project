#!/usr/bin/env python3
"""
Prepare Processed Test Data for Model Evaluation
Combines human and AI-generated articles with extracted features
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# Import feature extraction modules
try:
    from src.feature_engineering.stylometric_analysis import StylometricAnalyzer
except ImportError:
    StylometricAnalyzer = None
    print("Warning: StylometricAnalyzer not available")

try:
    from src.feature_engineering.semantic_fluctuation_analysis import SemanticFluctuationAnalyzer
except ImportError:
    SemanticFluctuationAnalyzer = None
    print("Warning: SemanticFluctuationAnalyzer not available")


class DataProcessor:
    """Process and prepare evaluation data"""

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize data processor

        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_human_data(self, data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load human-written articles

        Args:
            data_path: Path to human data JSON file
            max_samples: Maximum number of samples to load

        Returns:
            List of article dictionaries
        """
        print(f"Loading human data from {data_path}...")

        with open(data_path, 'r') as f:
            data = json.load(f)

        # Handle different data formats
        articles = []

        if isinstance(data, list):
            # Direct list of articles
            for item in data[:max_samples]:
                if isinstance(item, dict):
                    article = {
                        'text': item.get('text', item.get('article', item.get('content', ''))),
                        'source': 'human',
                        'label': 0,  # 0 for human
                        'id': item.get('id', f"human_{len(articles)}"),
                        'topic': item.get('topic', item.get('category', 'unknown'))
                    }
                    if article['text']:  # Only add if text exists
                        articles.append(article)

        elif isinstance(data, dict):
            # Dictionary with articles under a key
            article_list = data.get('articles', data.get('data', []))
            for item in article_list[:max_samples]:
                article = {
                    'text': item.get('text', item.get('article', '')),
                    'source': 'human',
                    'label': 0,
                    'id': item.get('id', f"human_{len(articles)}"),
                    'topic': item.get('topic', item.get('category', 'unknown'))
                }
                if article['text']:
                    articles.append(article)

        print(f"Loaded {len(articles)} human articles")
        return articles

    def load_ai_data(self, data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load AI-generated articles from directory

        Args:
            data_dir: Directory containing AI-generated JSON files
            max_samples: Maximum number of samples to load

        Returns:
            List of article dictionaries
        """
        print(f"Loading AI data from {data_dir}...")

        data_path = Path(data_dir)
        articles = []

        # Load all JSON files in the directory
        json_files = list(data_path.glob("*.json"))
        print(f"Found {len(json_files)} AI data files")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Extract model name from filename if possible
                filename = json_file.stem
                model_name = 'unknown'
                if 'gpt2' in filename.lower():
                    model_name = 'gpt2'
                elif 'gpt' in filename.lower():
                    model_name = 'gpt'
                elif 'claude' in filename.lower():
                    model_name = 'claude'
                elif 'mistral' in filename.lower():
                    model_name = 'mistral'
                elif 'llama' in filename.lower():
                    model_name = 'llama'

                # Process articles
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            article = {
                                'text': item.get('generated_text', item.get('text', item.get('content', ''))),
                                'source': 'ai',
                                'label': 1,  # 1 for AI
                                'model': item.get('model', model_name),
                                'id': item.get('id', f"ai_{model_name}_{len(articles)}"),
                                'topic': item.get('topic', item.get('category', 'unknown')),
                                'original_id': item.get('original_id', None)
                            }
                            if article['text'] and len(article['text']) > 50:  # Filter short texts
                                articles.append(article)

                elif isinstance(data, dict) and 'articles' in data:
                    for item in data['articles']:
                        article = {
                            'text': item.get('generated_text', item.get('text', '')),
                            'source': 'ai',
                            'label': 1,
                            'model': item.get('model', model_name),
                            'id': f"ai_{model_name}_{len(articles)}",
                            'topic': item.get('topic', 'unknown')
                        }
                        if article['text'] and len(article['text']) > 50:
                            articles.append(article)

                if max_samples and len(articles) >= max_samples:
                    break

            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        print(f"Loaded {len(articles)} AI articles")

        # Show distribution of models
        if articles:
            model_counts = {}
            for article in articles:
                model = article.get('model', 'unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            print("AI model distribution:")
            for model, count in model_counts.items():
                print(f"  {model}: {count}")

        return articles[:max_samples] if max_samples else articles

    def balance_dataset(self, human_articles: List[Dict], ai_articles: List[Dict]) -> List[Dict]:
        """
        Balance human and AI articles

        Args:
            human_articles: List of human articles
            ai_articles: List of AI articles

        Returns:
            Balanced list of articles
        """
        min_count = min(len(human_articles), len(ai_articles))
        print(f"\nBalancing dataset to {min_count} samples per class")

        # Sample to balance
        if len(human_articles) > min_count:
            human_articles = random.sample(human_articles, min_count)
        if len(ai_articles) > min_count:
            ai_articles = random.sample(ai_articles, min_count)

        # Combine and shuffle
        all_articles = human_articles + ai_articles
        random.shuffle(all_articles)

        return all_articles

    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic features from text for quick processing

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        words = text.split()
        sentences = text.split('.')

        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s]) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }

        return features

    def extract_advanced_features(self, articles: List[Dict],
                                 use_stylometric: bool = True,
                                 use_semantic: bool = False) -> List[Dict]:
        """
        Extract advanced features from articles

        Args:
            articles: List of article dictionaries
            use_stylometric: Whether to extract stylometric features
            use_semantic: Whether to extract semantic features (slower)

        Returns:
            Articles with extracted features
        """
        print("\nExtracting features...")

        # Initialize analyzers if needed
        stylo_analyzer = None
        semantic_analyzer = None

        if use_stylometric:
            if StylometricAnalyzer is not None:
                try:
                    stylo_analyzer = StylometricAnalyzer()
                    print("Stylometric analyzer initialized")
                except Exception as e:
                    print(f"Could not initialize stylometric analyzer: {e}")
                    use_stylometric = False
            else:
                print("StylometricAnalyzer not available, skipping")
                use_stylometric = False

        if use_semantic:
            if SemanticFluctuationAnalyzer is not None:
                try:
                    semantic_analyzer = SemanticFluctuationAnalyzer()
                    print("Semantic analyzer initialized")
                except Exception as e:
                    print(f"Could not initialize semantic analyzer: {e}")
                    use_semantic = False
            else:
                print("SemanticFluctuationAnalyzer not available, skipping")
                use_semantic = False

        # Extract features for each article
        for article in tqdm(articles, desc="Extracting features"):
            features = self.extract_basic_features(article['text'])

            # Add stylometric features
            if use_stylometric and stylo_analyzer:
                try:
                    stylo_features = stylo_analyzer.extract_features(article['text'])
                    features.update({f"stylo_{k}": v for k, v in stylo_features.items()})
                except Exception as e:
                    print(f"Error extracting stylometric features: {e}")

            # Add semantic features (optional, slower)
            if use_semantic and semantic_analyzer:
                try:
                    semantic_features = semantic_analyzer.extract_features(article['text'])
                    features.update({f"semantic_{k}": v for k, v in semantic_features.items()})
                except Exception as e:
                    print(f"Error extracting semantic features: {e}")

            article['features'] = features

        return articles

    def split_data(self, articles: List[Dict],
                   test_size: float = 0.2,
                   val_size: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train, validation, and test sets

        Args:
            articles: List of articles
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Train, validation, and test sets
        """
        # Get labels for stratification
        labels = [article['label'] for article in articles]

        # First split: train+val and test
        train_val, test = train_test_split(
            articles,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )

        # Second split: train and val
        train_val_labels = [article['label'] for article in train_val]
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val size

        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val_labels,
            random_state=42
        )

        print(f"\nData split:")
        print(f"  Train: {len(train)} samples")
        print(f"  Val: {len(val)} samples")
        print(f"  Test: {len(test)} samples")

        return train, val, test

    def save_processed_data(self,
                          train: List[Dict],
                          val: List[Dict],
                          test: List[Dict]):
        """
        Save processed data to files

        Args:
            train: Training data
            val: Validation data
            test: Test data
        """
        # Save as JSON
        train_path = self.output_dir / "train_data.json"
        val_path = self.output_dir / "val_data.json"
        test_path = self.output_dir / "test_data.json"

        with open(train_path, 'w') as f:
            json.dump(train, f, indent=2)
        print(f"Saved training data to {train_path}")

        with open(val_path, 'w') as f:
            json.dump(val, f, indent=2)
        print(f"Saved validation data to {val_path}")

        with open(test_path, 'w') as f:
            json.dump(test, f, indent=2)
        print(f"Saved test data to {test_path}")

        # Also save as CSV for easier inspection
        for data, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
            df_data = []
            for article in data:
                row = {
                    'id': article['id'],
                    'text': article['text'][:500],  # Truncate for CSV
                    'label': article['label'],
                    'source': article['source']
                }
                if 'model' in article:
                    row['model'] = article['model']
                if 'features' in article:
                    row.update(article['features'])
                df_data.append(row)

            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / f"{name}_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {name} CSV to {csv_path}")

        # Save metadata
        metadata = {
            'timestamp': self.timestamp,
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'train_human': sum(1 for a in train if a['label'] == 0),
            'train_ai': sum(1 for a in train if a['label'] == 1),
            'val_human': sum(1 for a in val if a['label'] == 0),
            'val_ai': sum(1 for a in val if a['label'] == 1),
            'test_human': sum(1 for a in test if a['label'] == 0),
            'test_ai': sum(1 for a in test if a['label'] == 1),
            'features_extracted': 'features' in train[0] if train else False
        }

        metadata_path = self.output_dir / "data_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved metadata to {metadata_path}")

    def process_pipeline(self,
                        human_data_path: str,
                        ai_data_dir: str,
                        max_samples: Optional[int] = None,
                        extract_features: bool = True,
                        use_semantic: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Complete processing pipeline

        Args:
            human_data_path: Path to human data
            ai_data_dir: Directory with AI data
            max_samples: Maximum samples per class
            extract_features: Whether to extract features
            use_semantic: Whether to use semantic features (slower)

        Returns:
            Train, validation, and test sets
        """
        print("="*80)
        print("DATA PROCESSING PIPELINE")
        print("="*80)

        # Load data
        human_articles = self.load_human_data(human_data_path, max_samples)
        ai_articles = self.load_ai_data(ai_data_dir, max_samples)

        # Balance dataset
        balanced_articles = self.balance_dataset(human_articles, ai_articles)
        print(f"\nTotal balanced samples: {len(balanced_articles)}")

        # Extract features
        if extract_features:
            balanced_articles = self.extract_advanced_features(
                balanced_articles,
                use_stylometric=True,
                use_semantic=use_semantic
            )

        # Split data
        train, val, test = self.split_data(balanced_articles)

        # Save processed data
        self.save_processed_data(train, val, test)

        print("\n" + "="*80)
        print("PROCESSING COMPLETE!")
        print("="*80)

        return train, val, test


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Prepare processed data for model evaluation'
    )

    parser.add_argument('--human_data', type=str,
                       default='data/datasets/combined_human_news_full.json',
                       help='Path to human data JSON file')
    parser.add_argument('--ai_data', type=str,
                       default='data/ai_generated',
                       help='Directory containing AI-generated data')
    parser.add_argument('--output_dir', type=str,
                       default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per class (None for all)')
    parser.add_argument('--no_features', action='store_true',
                       help='Skip feature extraction')
    parser.add_argument('--semantic', action='store_true',
                       help='Include semantic features (slower)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set proportion')

    args = parser.parse_args()

    # Initialize processor
    processor = DataProcessor(args.output_dir)

    # Run processing pipeline
    train, val, test = processor.process_pipeline(
        args.human_data,
        args.ai_data,
        max_samples=args.max_samples,
        extract_features=not args.no_features,
        use_semantic=args.semantic
    )

    print(f"\nProcessed data saved to: {args.output_dir}")
    print("\nYou can now run evaluation with:")
    print(f"python src/evaluation/run_evaluation.py \\")
    print(f"    --test_data {args.output_dir}/test_data.json \\")
    print(f"    --train_data {args.output_dir}/train_data.json \\")
    print(f"    --model_path models/ensemble_model.pkl")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Stylometric Analysis for AI-Generated Text Detection
Extracts linguistic and stylistic features from text for classification
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import string
import re
from tqdm import tqdm
import argparse

# NLP libraries for advanced features
try:
    import spacy
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from nltk.tree import Tree
    from scipy import stats

    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("Note: Install spacy and nltk for full feature extraction")
    print("Run: pip install spacy nltk scipy")
    print("Then: python -m spacy download en_core_web_sm")


class StylometricAnalyzer:
    """Extract stylometric features from text"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the analyzer

        Args:
            spacy_model: SpaCy model to use for parsing
        """
        self.feature_names = []

        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except:
                print(f"SpaCy model {spacy_model} not found. Installing...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", spacy_model])
                self.nlp = spacy.load(spacy_model)
        else:
            self.nlp = None

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all stylometric features from text

        Args:
            text: Input text

        Returns:
            Dictionary of feature names and values
        """
        features = {}

        # Clean text
        text = text.strip()
        if not text:
            return features

        # 1. Lexical Diversity Metrics
        lexical_features = self._extract_lexical_diversity(text)
        features.update(lexical_features)

        # 2. Sentence Complexity Patterns
        if self.nlp:
            complexity_features = self._extract_sentence_complexity(text)
            features.update(complexity_features)

        # 3. N-gram Frequency Distributions
        ngram_features = self._extract_ngram_patterns(text)
        features.update(ngram_features)

        # 4. Punctuation and Formatting Patterns
        punctuation_features = self._extract_punctuation_patterns(text)
        features.update(punctuation_features)

        # 5. Additional Statistical Features
        statistical_features = self._extract_statistical_features(text)
        features.update(statistical_features)

        return features

    def _extract_lexical_diversity(self, text: str) -> Dict[str, float]:
        """Extract lexical diversity metrics"""
        features = {}

        # Tokenize
        words = word_tokenize(text.lower()) if NLP_AVAILABLE else text.lower().split()
        words = [w for w in words if w.isalpha()]  # Keep only alphabetic tokens

        if len(words) == 0:
            return features

        # Type-Token Ratio (TTR)
        unique_words = set(words)
        features['ttr'] = len(unique_words) / len(words) if words else 0

        # Hapax Legomena (words appearing only once)
        word_freq = Counter(words)
        hapax = [w for w, c in word_freq.items() if c == 1]
        features['hapax_ratio'] = len(hapax) / len(words) if words else 0

        # Dis Legomena (words appearing twice)
        dis = [w for w, c in word_freq.items() if c == 2]
        features['dis_ratio'] = len(dis) / len(words) if words else 0

        # Simpson's Diversity Index
        N = len(words)
        simpson = sum(n * (n - 1) for n in word_freq.values())
        features['simpson_index'] = 1 - (simpson / (N * (N - 1))) if N > 1 else 0

        # Yule's K (vocabulary richness)
        M1 = len(words)
        M2 = sum(freq ** 2 for freq in word_freq.values())
        features['yule_k'] = 10000 * (M2 - M1) / (M1 ** 2) if M1 > 0 else 0

        # Average word length
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0

        # Vocabulary size
        features['vocabulary_size'] = len(unique_words)

        return features

    def _extract_sentence_complexity(self, text: str) -> Dict[str, float]:
        """Extract sentence complexity features using SpaCy"""
        features = {}

        if not self.nlp:
            return features

        # Parse with SpaCy
        doc = self.nlp(text[:1000000])  # Limit text length for SpaCy

        # Sentence-level features
        sentences = list(doc.sents)
        if sentences:
            # Average sentence length
            sent_lengths = [len(sent) for sent in sentences]
            features['avg_sent_length'] = np.mean(sent_lengths)
            features['std_sent_length'] = np.std(sent_lengths)

            # Parse tree depth
            depths = []
            for sent in sentences:
                depth = self._get_tree_depth(sent.root)
                depths.append(depth)
            features['avg_tree_depth'] = np.mean(depths) if depths else 0
            features['max_tree_depth'] = max(depths) if depths else 0

            # Dependency distances
            dep_distances = []
            for token in doc:
                if token.dep_ != "ROOT" and token.head:
                    distance = abs(token.i - token.head.i)
                    dep_distances.append(distance)
            features['avg_dep_distance'] = np.mean(dep_distances) if dep_distances else 0
            features['max_dep_distance'] = max(dep_distances) if dep_distances else 0

        # Syntactic features
        pos_tags = [token.pos_ for token in doc]
        if pos_tags:
            pos_counts = Counter(pos_tags)
            total_pos = len(pos_tags)

            # POS tag ratios
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PRON']:
                features[f'pos_{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total_pos

        # Named entity density
        entities = doc.ents
        features['entity_density'] = len(entities) / len(doc) if len(doc) > 0 else 0

        return features

    def _get_tree_depth(self, token, depth: int = 0) -> int:
        """Calculate parse tree depth recursively"""
        if not list(token.children):
            return depth
        return max(self._get_tree_depth(child, depth + 1) for child in token.children)

    def _extract_ngram_patterns(self, text: str) -> Dict[str, float]:
        """Extract n-gram frequency patterns"""
        features = {}

        # Tokenize
        words = word_tokenize(text.lower()) if NLP_AVAILABLE else text.lower().split()
        words = [w for w in words if w.isalpha()]

        if not words:
            return features

        # Character n-grams (for style)
        text_clean = ''.join([c for c in text.lower() if c.isalpha() or c.isspace()])

        # Bigram and trigram frequencies
        for n in [2, 3]:
            if len(words) >= n:
                n_grams = list(ngrams(words, n)) if NLP_AVAILABLE else []
                if n_grams:
                    n_gram_freq = Counter(n_grams)
                    # Entropy of n-gram distribution
                    total = sum(n_gram_freq.values())
                    probs = [count / total for count in n_gram_freq.values()]
                    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                    features[f'{n}gram_entropy'] = entropy

                    # Unique n-gram ratio
                    features[f'{n}gram_unique_ratio'] = len(n_gram_freq) / len(n_grams)

        # Function words frequency
        function_words = ['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an',
                         'as', 'are', 'was', 'been', 'be', 'have', 'had']
        func_count = sum(1 for w in words if w in function_words)
        features['function_word_ratio'] = func_count / len(words) if words else 0

        return features

    def _extract_punctuation_patterns(self, text: str) -> Dict[str, float]:
        """Extract punctuation and formatting patterns"""
        features = {}

        if not text:
            return features

        # Punctuation counts
        punct_counts = Counter(c for c in text if c in string.punctuation)
        total_chars = len(text)

        # Individual punctuation ratios
        for punct in ['.', ',', '!', '?', ';', ':', '"', "'", '-', '(', ')']:
            features[f'punct_{punct}_ratio'] = punct_counts.get(punct, 0) / total_chars

        # Overall punctuation density
        total_punct = sum(punct_counts.values())
        features['punct_density'] = total_punct / total_chars

        # Sentence endings
        features['exclamation_ratio'] = text.count('!') / (text.count('.') + 1)
        features['question_ratio'] = text.count('?') / (text.count('.') + 1)

        # Capitalization patterns
        words = text.split()
        if words:
            # Ratio of capitalized words (excluding sentence starts)
            cap_words = sum(1 for w in words[1:] if w and w[0].isupper())
            features['capitalized_ratio'] = cap_words / len(words)

            # All caps words
            all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            features['all_caps_ratio'] = all_caps / len(words)

        # Whitespace patterns
        features['space_ratio'] = text.count(' ') / total_chars
        features['newline_ratio'] = text.count('\n') / total_chars

        # Digit usage
        digit_count = sum(1 for c in text if c.isdigit())
        features['digit_ratio'] = digit_count / total_chars

        return features

    def _extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Extract additional statistical text features"""
        features = {}

        # Sentence statistics
        sentences = sent_tokenize(text) if NLP_AVAILABLE else text.split('.')
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sent_lengths:
                features['sent_length_mean'] = np.mean(sent_lengths)
                features['sent_length_std'] = np.std(sent_lengths)
                features['sent_length_min'] = min(sent_lengths)
                features['sent_length_max'] = max(sent_lengths)
                features['sent_count'] = len(sent_lengths)

        # Word length distribution
        words = text.split()
        if words:
            word_lengths = [len(w) for w in words]
            features['word_length_mean'] = np.mean(word_lengths)
            features['word_length_std'] = np.std(word_lengths)
            features['word_length_skew'] = stats.skew(word_lengths) if len(word_lengths) > 2 else 0
            features['word_length_kurtosis'] = stats.kurtosis(word_lengths) if len(word_lengths) > 3 else 0

            # Short vs long words
            features['short_word_ratio'] = sum(1 for l in word_lengths if l <= 3) / len(words)
            features['long_word_ratio'] = sum(1 for l in word_lengths if l >= 10) / len(words)

        # Readability scores (simplified)
        if words and sentences:
            # Flesch Reading Ease approximation
            total_words = len(words)
            total_sentences = len(sentences)
            total_syllables = sum(self._count_syllables(w) for w in words)

            if total_sentences > 0:
                asl = total_words / total_sentences  # Average Sentence Length
                asw = total_syllables / total_words if total_words > 0 else 0  # Average Syllables per Word
                features['flesch_reading_ease'] = 206.835 - 1.015 * asl - 84.6 * asw
                features['flesch_kincaid_grade'] = 0.39 * asl + 11.8 * asw - 15.59

        return features

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter"""
        word = word.lower()
        vowels = 'aeiou'
        syllables = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel

        if word.endswith('e'):
            syllables -= 1
        if syllables == 0:
            syllables = 1

        return syllables


def process_dataset(input_file: str, output_dir: str, sample_size: int = None):
    """
    Process a dataset and extract stylometric features

    Args:
        input_file: Path to input JSON file
        output_dir: Output directory for features
        sample_size: Optional limit for testing
    """
    # Initialize analyzer
    analyzer = StylometricAnalyzer()

    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        samples = data
    elif 'train' in data and 'test' in data:
        samples = data['train'] + data['test']
    elif 'articles' in data:
        samples = data['articles']
    else:
        print("Unexpected JSON structure")
        return

    if sample_size:
        samples = samples[:sample_size]

    print(f"Processing {len(samples)} samples...")

    # Extract features
    features_list = []
    labels = []

    for sample in tqdm(samples, desc="Extracting features"):
        # Get text
        if isinstance(sample, dict):
            text = sample.get('text', sample.get('article', sample.get('document', '')))
            label = sample.get('label', sample.get('is_ai', -1))
        else:
            text = str(sample)
            label = -1

        if not text:
            continue

        # Extract features
        features = analyzer.extract_features(text)
        features_list.append(features)
        labels.append(label)

    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    df_features['label'] = labels

    # Fill missing values with 0
    df_features = df_features.fillna(0)

    # Save features
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_file = output_path / f"stylometric_features_{Path(input_file).stem}.csv"
    df_features.to_csv(csv_file, index=False)
    print(f"Saved features to {csv_file}")

    # Save as numpy arrays for ML
    X = df_features.drop('label', axis=1).values
    y = df_features['label'].values

    np_file = output_path / f"stylometric_features_{Path(input_file).stem}.npz"
    np.savez(np_file, X=X, y=y, feature_names=df_features.columns[:-1].tolist())
    print(f"Saved numpy arrays to {np_file}")

    # Print feature statistics
    print(f"\nFeature Statistics:")
    print(f"  Number of samples: {len(df_features)}")
    print(f"  Number of features: {len(df_features.columns) - 1}")
    print(f"  Label distribution: {Counter(labels)}")

    # Print top features by variance
    feature_vars = df_features.drop('label', axis=1).var().sort_values(ascending=False)
    print(f"\nTop 10 features by variance:")
    for feat, var in feature_vars.head(10).items():
        print(f"  {feat}: {var:.4f}")

    return df_features


def main():
    parser = argparse.ArgumentParser(description='Extract stylometric features')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with text data')
    parser.add_argument('--output', type=str, default='../../features',
                       help='Output directory for features')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')

    args = parser.parse_args()

    # Process dataset
    process_dataset(args.input, args.output, args.sample)


if __name__ == "__main__":
    # If no arguments, run test
    import sys
    if len(sys.argv) == 1:
        print("Running test with sample data...")

        # Create test data
        test_data = [
            {
                "text": "This is a test article. It has multiple sentences. The sentences vary in length.",
                "label": 0
            },
            {
                "text": "AI generated text often follows patterns. Patterns can be detected. Detection is important.",
                "label": 1
            }
        ]

        # Save test data
        with open('/tmp/test_stylometric.json', 'w') as f:
            json.dump(test_data, f)

        # Process
        df = process_dataset('/tmp/test_stylometric.json', '/tmp/')
        print("\nTest completed successfully!")
    else:
        main()
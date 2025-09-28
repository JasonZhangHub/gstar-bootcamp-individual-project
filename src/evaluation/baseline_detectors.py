#!/usr/bin/env python3
"""
Baseline AI Text Detection Methods for Benchmarking
Implements various existing approaches for comparison
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import warnings
from tqdm import tqdm

# ML libraries
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Deep learning
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Some baselines will be skipped.")

# Statistical methods
from scipy import stats
import math

warnings.filterwarnings('ignore')


class PerplexityDetector(BaseEstimator, ClassifierMixin):
    """
    Perplexity-based detection (inspired by DetectGPT)
    Lower perplexity often indicates AI-generated text
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold
        self.perplexity_scores_ = None

    def calculate_perplexity(self, texts: List[str]) -> np.ndarray:
        """
        Calculate perplexity scores for texts
        Simplified version - in practice would use language model
        """
        perplexities = []

        for text in texts:
            # Simplified perplexity calculation
            words = text.split()
            vocab_size = len(set(words))
            text_length = len(words)

            # Estimate entropy (simplified)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            entropy = 0
            for count in word_counts.values():
                prob = count / text_length
                if prob > 0:
                    entropy -= prob * math.log2(prob)

            # Approximate perplexity
            perplexity = 2 ** entropy
            perplexities.append(perplexity)

        return np.array(perplexities)

    def fit(self, X: List[str], y: np.ndarray):
        """Fit the detector by finding optimal threshold"""
        self.perplexity_scores_ = self.calculate_perplexity(X)

        if self.threshold is None:
            # Find optimal threshold using training data
            human_perp = self.perplexity_scores_[y == 0]
            ai_perp = self.perplexity_scores_[y == 1]

            # Set threshold as mean between distributions
            self.threshold = (np.mean(human_perp) + np.mean(ai_perp)) / 2

        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict if text is AI-generated based on perplexity"""
        perplexities = self.calculate_perplexity(X)
        # Lower perplexity suggests AI-generated
        predictions = (perplexities < self.threshold).astype(int)
        return predictions

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get probability estimates"""
        perplexities = self.calculate_perplexity(X)

        # Convert perplexity to probability (sigmoid-like)
        # Lower perplexity -> higher AI probability
        max_perp = np.max(perplexities)
        min_perp = np.min(perplexities)
        normalized = (perplexities - min_perp) / (max_perp - min_perp + 1e-8)

        # Invert so lower perplexity gives higher AI probability
        ai_probs = 1 - normalized
        human_probs = normalized

        return np.column_stack([human_probs, ai_probs])


class BurstinessDetector(BaseEstimator, ClassifierMixin):
    """
    Burstiness-based detection
    Measures variance in sentence lengths and word frequencies
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold
        self.scaler = StandardScaler()

    def calculate_burstiness(self, texts: List[str]) -> np.ndarray:
        """Calculate burstiness features for texts"""
        features = []

        for text in texts:
            sentences = text.split('.')
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

            if len(sentence_lengths) > 1:
                # Sentence length variance
                sent_var = np.var(sentence_lengths)
                sent_mean = np.mean(sentence_lengths)
                sent_cv = sent_var / (sent_mean + 1e-8)  # Coefficient of variation
            else:
                sent_var = sent_mean = sent_cv = 0

            # Word frequency burstiness
            words = text.lower().split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            if word_counts:
                frequencies = list(word_counts.values())
                freq_var = np.var(frequencies)
                freq_mean = np.mean(frequencies)
                freq_cv = freq_var / (freq_mean + 1e-8)
            else:
                freq_var = freq_mean = freq_cv = 0

            features.append([sent_var, sent_cv, freq_var, freq_cv])

        return np.array(features)

    def fit(self, X: List[str], y: np.ndarray):
        """Fit the detector"""
        features = self.calculate_burstiness(X)
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)

        # Use simple threshold on combined score
        scores = np.mean(features_scaled, axis=1)

        if self.threshold is None:
            human_scores = scores[y == 0]
            ai_scores = scores[y == 1]
            self.threshold = (np.mean(human_scores) + np.mean(ai_scores)) / 2

        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict based on burstiness"""
        features = self.calculate_burstiness(X)
        features_scaled = self.scaler.transform(features)
        scores = np.mean(features_scaled, axis=1)

        # Higher burstiness suggests human text
        predictions = (scores < self.threshold).astype(int)
        return predictions

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get probability estimates"""
        features = self.calculate_burstiness(X)
        features_scaled = self.scaler.transform(features)
        scores = np.mean(features_scaled, axis=1)

        # Sigmoid transformation
        ai_probs = 1 / (1 + np.exp(scores - self.threshold))
        human_probs = 1 - ai_probs

        return np.column_stack([human_probs, ai_probs])


class GLTRDetector(BaseEstimator, ClassifierMixin):
    """
    GLTR-inspired detector
    Analyzes token probability distributions
    """

    def __init__(self, n_bins: int = 4):
        self.n_bins = n_bins
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = LogisticRegression(random_state=42)

    def extract_gltr_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract GLTR-like features
        Simplified version without actual language model
        """
        features = []

        for text in texts:
            words = text.split()
            total_words = len(words)

            if total_words == 0:
                features.append([0] * self.n_bins)
                continue

            # Simulate probability bins (would use actual LM in practice)
            # Count word frequencies as proxy
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Sort by frequency
            sorted_counts = sorted(word_counts.values(), reverse=True)

            # Distribute into bins
            bin_counts = [0] * self.n_bins
            for i, count in enumerate(sorted_counts):
                bin_idx = min(i * self.n_bins // len(sorted_counts), self.n_bins - 1)
                bin_counts[bin_idx] += count

            # Normalize
            bin_probs = [c / total_words for c in bin_counts]
            features.append(bin_probs)

        return np.array(features)

    def fit(self, X: List[str], y: np.ndarray):
        """Fit the GLTR detector"""
        # Extract GLTR features
        gltr_features = self.extract_gltr_features(X)

        # Also use TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(X).toarray()

        # Combine features
        combined_features = np.hstack([gltr_features, tfidf_features])

        # Train classifier
        self.classifier.fit(combined_features, y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict using GLTR features"""
        gltr_features = self.extract_gltr_features(X)
        tfidf_features = self.vectorizer.transform(X).toarray()
        combined_features = np.hstack([gltr_features, tfidf_features])
        return self.classifier.predict(combined_features)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get probability estimates"""
        gltr_features = self.extract_gltr_features(X)
        tfidf_features = self.vectorizer.transform(X).toarray()
        combined_features = np.hstack([gltr_features, tfidf_features])
        return self.classifier.predict_proba(combined_features)


class NgramDetector(BaseEstimator, ClassifierMixin):
    """
    N-gram based detector
    Uses character and word n-grams with machine learning
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)):
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=500
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            max_features=500
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def fit(self, X: List[str], y: np.ndarray):
        """Fit the n-gram detector"""
        # Extract n-gram features
        char_features = self.char_vectorizer.fit_transform(X).toarray()
        word_features = self.word_vectorizer.fit_transform(X).toarray()

        # Combine features
        combined_features = np.hstack([char_features, word_features])

        # Train classifier
        self.classifier.fit(combined_features, y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict using n-gram features"""
        char_features = self.char_vectorizer.transform(X).toarray()
        word_features = self.word_vectorizer.transform(X).toarray()
        combined_features = np.hstack([char_features, word_features])
        return self.classifier.predict(combined_features)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get probability estimates"""
        char_features = self.char_vectorizer.transform(X).toarray()
        word_features = self.word_vectorizer.transform(X).toarray()
        combined_features = np.hstack([char_features, word_features])
        return self.classifier.predict_proba(combined_features)


class SimpleTransformerDetector(BaseEstimator, ClassifierMixin):
    """
    Simple transformer-based detector using pre-trained models
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = LogisticRegression(random_state=42)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from transformer model"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            ).to(self.device)

        embeddings = []

        for text in tqdm(texts, desc="Extracting embeddings"):
            # Tokenize and truncate
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use [CLS] token embedding
                embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                embeddings.append(embedding.flatten())

        return np.array(embeddings)

    def fit(self, X: List[str], y: np.ndarray):
        """Fit the transformer detector"""
        print("Extracting training embeddings...")
        embeddings = self.extract_embeddings(X)
        print("Training classifier...")
        self.classifier.fit(embeddings, y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict using transformer embeddings"""
        embeddings = self.extract_embeddings(X)
        return self.classifier.predict(embeddings)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get probability estimates"""
        embeddings = self.extract_embeddings(X)
        return self.classifier.predict_proba(embeddings)


class ZipfianDetector(BaseEstimator, ClassifierMixin):
    """
    Zipfian distribution-based detector
    Analyzes word frequency distributions
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold

    def calculate_zipf_coefficient(self, text: str) -> float:
        """Calculate Zipf coefficient for text"""
        words = text.lower().split()

        if len(words) == 0:
            return 0

        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency
        frequencies = sorted(word_counts.values(), reverse=True)

        if len(frequencies) < 2:
            return 0

        # Fit Zipf distribution (simplified)
        # log(frequency) = -s * log(rank) + c
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(np.array(frequencies) + 1)  # Add 1 to avoid log(0)

        # Linear regression
        coefficient = np.polyfit(log_ranks, log_freqs, 1)[0]

        return abs(coefficient)

    def fit(self, X: List[str], y: np.ndarray):
        """Fit the Zipfian detector"""
        coefficients = [self.calculate_zipf_coefficient(text) for text in X]

        if self.threshold is None:
            human_coeffs = [c for c, label in zip(coefficients, y) if label == 0]
            ai_coeffs = [c for c, label in zip(coefficients, y) if label == 1]
            self.threshold = (np.mean(human_coeffs) + np.mean(ai_coeffs)) / 2

        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict based on Zipfian distribution"""
        coefficients = [self.calculate_zipf_coefficient(text) for text in X]
        # AI text often has more regular Zipfian distribution
        predictions = (np.array(coefficients) > self.threshold).astype(int)
        return predictions

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get probability estimates"""
        coefficients = np.array([self.calculate_zipf_coefficient(text) for text in X])

        # Sigmoid transformation
        ai_probs = 1 / (1 + np.exp(-(coefficients - self.threshold)))
        human_probs = 1 - ai_probs

        return np.column_stack([human_probs, ai_probs])


def get_baseline_detectors() -> Dict[str, BaseEstimator]:
    """
    Get dictionary of baseline detectors

    Returns:
        Dictionary mapping detector names to instances
    """
    detectors = {
        'Perplexity': PerplexityDetector(),
        'Burstiness': BurstinessDetector(),
        'GLTR': GLTRDetector(),
        'N-gram': NgramDetector(),
        'Zipfian': ZipfianDetector()
    }

    # Add transformer detector if available
    if TRANSFORMERS_AVAILABLE:
        try:
            detectors['Transformer'] = SimpleTransformerDetector()
        except Exception as e:
            print(f"Could not initialize transformer detector: {e}")

    return detectors


def train_baseline_detectors(X_train: List[str],
                            y_train: np.ndarray,
                            detectors: Optional[Dict[str, BaseEstimator]] = None) -> Dict[str, BaseEstimator]:
    """
    Train baseline detectors

    Args:
        X_train: Training texts
        y_train: Training labels (0=human, 1=AI)
        detectors: Optional dictionary of detectors (uses defaults if None)

    Returns:
        Dictionary of trained detectors
    """
    if detectors is None:
        detectors = get_baseline_detectors()

    trained_detectors = {}

    for name, detector in detectors.items():
        print(f"Training {name} detector...")
        try:
            detector.fit(X_train, y_train)
            trained_detectors[name] = detector
            print(f"  ✓ {name} trained successfully")
        except Exception as e:
            print(f"  ✗ Error training {name}: {e}")

    return trained_detectors


def evaluate_baselines(X_test: List[str],
                      y_test: np.ndarray,
                      detectors: Dict[str, BaseEstimator]) -> pd.DataFrame:
    """
    Evaluate baseline detectors

    Args:
        X_test: Test texts
        y_test: Test labels
        detectors: Dictionary of trained detectors

    Returns:
        DataFrame with evaluation results
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = []

    for name, detector in detectors.items():
        print(f"Evaluating {name}...")
        try:
            y_pred = detector.predict(X_test)

            result = {
                'detector': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }

            results.append(result)
        except Exception as e:
            print(f"  Error evaluating {name}: {e}")

    return pd.DataFrame(results)


def main():
    """Example usage of baseline detectors"""
    # Example data
    human_texts = [
        "The weather today is quite pleasant, with a gentle breeze.",
        "I went to the store yesterday and bought some groceries."
    ]
    ai_texts = [
        "The implementation of neural networks has revolutionized artificial intelligence.",
        "Machine learning algorithms process vast amounts of data efficiently."
    ]

    X_train = human_texts + ai_texts
    y_train = np.array([0, 0, 1, 1])  # 0=human, 1=AI

    # Train baselines
    detectors = train_baseline_detectors(X_train, y_train)

    # Evaluate
    results = evaluate_baselines(X_train, y_train, detectors)
    print("\nBaseline Results:")
    print(results)


if __name__ == "__main__":
    main()
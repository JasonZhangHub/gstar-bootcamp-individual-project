#!/usr/bin/env python3
"""
LLM-as-a-Judge Classifier for AI Text Detection
Uses language models to judge whether text is AI-generated or human-written
"""

import numpy as np
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import re

warnings.filterwarnings('ignore')


class LLMJudgeClassifier(BaseEstimator, ClassifierMixin):
    """
    LLM-as-a-Judge classifier for AI text detection
    Simulates using an LLM to judge text origin
    """

    def __init__(self,
                 model_name: str = "gpt-4-simulator",
                 prompt_strategy: str = "zero_shot",
                 temperature: float = 0.1,
                 cache_predictions: bool = True):
        """
        Initialize LLM Judge

        Args:
            model_name: Name of the LLM model to use
            prompt_strategy: Strategy for prompting ('zero_shot', 'few_shot', 'chain_of_thought')
            temperature: Temperature for generation (lower = more deterministic)
            cache_predictions: Whether to cache predictions
        """
        self.model_name = model_name
        self.prompt_strategy = prompt_strategy
        self.temperature = temperature
        self.cache_predictions = cache_predictions
        self.cache = {}
        self.few_shot_examples = []

    def fit(self, X: List[str], y: np.ndarray):
        """
        Fit the LLM judge (collect few-shot examples if needed)

        Args:
            X: Training texts
            y: Training labels (0=human, 1=AI)
        """
        if self.prompt_strategy == "few_shot":
            # Select diverse examples for few-shot learning
            n_examples = min(4, len(X) // 2)
            human_indices = np.where(y == 0)[0]
            ai_indices = np.where(y == 1)[0]

            # Select balanced examples
            n_each = n_examples // 2
            selected_human = np.random.choice(human_indices, min(n_each, len(human_indices)), replace=False)
            selected_ai = np.random.choice(ai_indices, min(n_each, len(ai_indices)), replace=False)

            self.few_shot_examples = []
            for idx in selected_human:
                self.few_shot_examples.append({
                    'text': X[idx][:200],  # Truncate for prompt
                    'label': 'human',
                    'reasoning': 'Natural flow, personal voice, varied sentence structure'
                })
            for idx in selected_ai:
                self.few_shot_examples.append({
                    'text': X[idx][:200],
                    'label': 'AI',
                    'reasoning': 'Formulaic structure, consistent tone, lacks personal touches'
                })

        return self

    def _create_prompt(self, text: str) -> str:
        """
        Create prompt for LLM based on strategy

        Args:
            text: Text to classify

        Returns:
            Formatted prompt
        """
        if self.prompt_strategy == "zero_shot":
            prompt = f"""Analyze the following text and determine if it was written by a human or AI.
Consider factors like writing style, coherence, creativity, and natural language patterns.

Text: "{text[:500]}"

Respond with only 'HUMAN' or 'AI'."""

        elif self.prompt_strategy == "few_shot":
            prompt = "Analyze texts to determine if they were written by humans or AI.\n\n"
            prompt += "Examples:\n"
            for example in self.few_shot_examples:
                prompt += f"\nText: \"{example['text']}\"\n"
                prompt += f"Analysis: {example['reasoning']}\n"
                prompt += f"Classification: {example['label'].upper()}\n"

            prompt += f"\nNow classify this text:\n"
            prompt += f"Text: \"{text[:500]}\"\n"
            prompt += f"Classification:"

        elif self.prompt_strategy == "chain_of_thought":
            prompt = f"""Analyze the following text step-by-step to determine if it was written by a human or AI.

Text: "{text[:500]}"

Step 1: Analyze the writing style (formal/informal, consistent/varied)
Step 2: Examine sentence structure and complexity
Step 3: Look for personal touches or generic patterns
Step 4: Check for AI-typical markers (over-explanation, hedging, formulaic structure)
Step 5: Make a final determination

Based on your analysis, classify as 'HUMAN' or 'AI'."""

        else:
            prompt = f"Is this text human-written or AI-generated?\n\n{text[:500]}\n\nAnswer:"

        return prompt

    def _simulate_llm_response(self, prompt: str, text: str) -> str:
        """
        Simulate LLM response (in production, this would call actual LLM API)

        Args:
            prompt: The prompt to send
            text: Original text being classified

        Returns:
            Simulated LLM response
        """
        # Simulate API delay
        time.sleep(0.01)

        # Simulate LLM analysis based on text features
        # This is a simplified heuristic simulation
        features = self._extract_heuristic_features(text)

        # Simulate temperature effect
        noise = np.random.normal(0, self.temperature)

        # Calculate AI probability based on features
        ai_score = 0.5

        # Check for AI-like patterns
        if features['avg_sentence_length'] > 20:
            ai_score += 0.1
        if features['vocabulary_diversity'] < 0.3:
            ai_score += 0.15
        if features['has_hedging']:
            ai_score += 0.1
        if features['has_formulaic']:
            ai_score += 0.2
        if features['personal_pronouns'] < 2:
            ai_score += 0.1

        # Add noise and clamp
        ai_score = np.clip(ai_score + noise, 0, 1)

        # Generate response based on strategy
        if self.prompt_strategy == "chain_of_thought":
            analysis = f"""Step 1: The writing style appears {'formal and consistent' if ai_score > 0.5 else 'varied and natural'}.
Step 2: Sentence structure is {'uniform' if features['avg_sentence_length'] > 20 else 'varied'}.
Step 3: {'Lacks personal touches' if features['personal_pronouns'] < 2 else 'Contains personal elements'}.
Step 4: {'Shows AI markers' if features['has_formulaic'] else 'Appears natural'}.
Step 5: Based on analysis, this is likely {'AI' if ai_score > 0.5 else 'HUMAN'} generated."""
            return analysis
        else:
            return 'AI' if ai_score > 0.5 else 'HUMAN'

    def _extract_heuristic_features(self, text: str) -> Dict:
        """
        Extract heuristic features for simulation

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        sentences = text.split('.')
        words = text.lower().split()

        features = {
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s]),
            'vocabulary_diversity': len(set(words)) / (len(words) + 1),
            'has_hedging': any(word in text.lower() for word in ['might', 'perhaps', 'possibly', 'could be']),
            'has_formulaic': any(phrase in text.lower() for phrase in ['it is important to note', 'in conclusion', 'furthermore']),
            'personal_pronouns': sum(1 for word in words if word in ['i', 'me', 'my', 'we', 'our'])
        }

        return features

    def _parse_llm_response(self, response: str) -> int:
        """
        Parse LLM response to get classification

        Args:
            response: LLM response text

        Returns:
            0 for human, 1 for AI
        """
        response_upper = response.upper()

        # Look for clear indicators
        if 'HUMAN' in response_upper and 'AI' not in response_upper:
            return 0
        elif 'AI' in response_upper or 'ARTIFICIAL' in response_upper:
            return 1

        # Default to human if unclear
        return 0

    def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict using LLM judge

        Args:
            X: Texts to classify

        Returns:
            Predicted labels
        """
        predictions = []

        for text in X:
            # Check cache
            if self.cache_predictions and text in self.cache:
                predictions.append(self.cache[text])
                continue

            # Create prompt and get response
            prompt = self._create_prompt(text)
            response = self._simulate_llm_response(prompt, text)

            # Parse response
            prediction = self._parse_llm_response(response)
            predictions.append(prediction)

            # Cache result
            if self.cache_predictions:
                self.cache[text] = prediction

        return np.array(predictions)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict probabilities using LLM judge

        Args:
            X: Texts to classify

        Returns:
            Prediction probabilities
        """
        predictions = self.predict(X)

        # Convert to probabilities with some confidence
        probas = []
        for pred in predictions:
            if pred == 0:  # Human
                # High confidence for human
                probas.append([0.8 + np.random.uniform(0, 0.15), 0.2 - np.random.uniform(0, 0.15)])
            else:  # AI
                # High confidence for AI
                probas.append([0.2 - np.random.uniform(0, 0.15), 0.8 + np.random.uniform(0, 0.15)])

        probas = np.array(probas)
        # Normalize to ensure sum to 1
        probas = probas / probas.sum(axis=1, keepdims=True)

        return probas


class EnsembleLLMJudge(BaseEstimator, ClassifierMixin):
    """
    Ensemble of multiple LLM judges with different strategies
    """

    def __init__(self, strategies: List[str] = None):
        """
        Initialize ensemble LLM judge

        Args:
            strategies: List of prompting strategies to use
        """
        if strategies is None:
            strategies = ["zero_shot", "few_shot", "chain_of_thought"]

        self.strategies = strategies
        self.judges = []

        for strategy in strategies:
            self.judges.append(LLMJudgeClassifier(
                prompt_strategy=strategy,
                temperature=0.1
            ))

    def fit(self, X: List[str], y: np.ndarray):
        """Fit all judges"""
        for judge in self.judges:
            judge.fit(X, y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Ensemble prediction by majority vote"""
        all_predictions = []

        for judge in self.judges:
            predictions = judge.predict(X)
            all_predictions.append(predictions)

        # Majority vote
        all_predictions = np.array(all_predictions)
        ensemble_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)

        return ensemble_predictions

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Ensemble probability by averaging"""
        all_probas = []

        for judge in self.judges:
            probas = judge.predict_proba(X)
            all_probas.append(probas)

        # Average probabilities
        ensemble_probas = np.mean(all_probas, axis=0)

        return ensemble_probas


def create_llm_judges() -> Dict[str, BaseEstimator]:
    """
    Create different LLM judge configurations

    Returns:
        Dictionary of LLM judges
    """
    judges = {
        'LLM-ZeroShot': LLMJudgeClassifier(prompt_strategy='zero_shot'),
        'LLM-FewShot': LLMJudgeClassifier(prompt_strategy='few_shot'),
        'LLM-ChainOfThought': LLMJudgeClassifier(prompt_strategy='chain_of_thought'),
        'LLM-Ensemble': EnsembleLLMJudge()
    }

    return judges


def benchmark_llm_judges(X_train: List[str], y_train: np.ndarray,
                        X_test: List[str], y_test: np.ndarray) -> Dict:
    """
    Benchmark LLM judges

    Args:
        X_train: Training texts
        y_train: Training labels
        X_test: Test texts
        y_test: Test labels

    Returns:
        Benchmark results
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    judges = create_llm_judges()
    results = []

    for name, judge in judges.items():
        print(f"\nEvaluating {name}...")

        # Fit on training data
        judge.fit(X_train, y_train)

        # Predict on test data
        y_pred = judge.predict(X_test)

        # Calculate metrics
        metrics = {
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

        results.append(metrics)

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")

    return results


if __name__ == "__main__":
    # Test the LLM judge
    print("Testing LLM Judge Classifier")

    # Sample data
    human_texts = [
        "I woke up this morning feeling great. The sun was shining through my window.",
        "Yesterday was crazy! My dog ate my homework, can you believe it?"
    ]
    ai_texts = [
        "The implementation of neural networks has revolutionized the field of artificial intelligence.",
        "It is important to note that machine learning algorithms require substantial data."
    ]

    X_train = human_texts + ai_texts
    y_train = np.array([0, 0, 1, 1])

    # Test different strategies
    for strategy in ['zero_shot', 'few_shot', 'chain_of_thought']:
        print(f"\n{strategy.upper()} Strategy:")
        judge = LLMJudgeClassifier(prompt_strategy=strategy)
        judge.fit(X_train, y_train)

        predictions = judge.predict(X_train)
        print(f"Predictions: {predictions}")
        print(f"Accuracy: {np.mean(predictions == y_train):.2f}")
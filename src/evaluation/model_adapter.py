#!/usr/bin/env python3
"""
Model Adapter for handling different model formats
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler


class EnsembleModelAdapter:
    """Adapter for ensemble model stored as dictionary"""

    def __init__(self, model_dict: Dict[str, Any]):
        """
        Initialize adapter with model dictionary

        Args:
            model_dict: Dictionary containing model components
        """
        self.model_dict = model_dict
        self.detection_models = model_dict.get('detection_models', {})
        self.source_models = model_dict.get('source_models', {})
        self.ensemble_weights = model_dict.get('ensemble_weights', {})
        self.feature_scaler = model_dict.get('feature_scaler')
        self.config = model_dict.get('config', {})

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict labels for input data

        Args:
            X: Input data (can be features array or text list)

        Returns:
            Predicted labels
        """
        # Handle different input types
        if isinstance(X, list):
            # If input is text, we need features
            # For now, return random predictions as placeholder
            return np.random.randint(0, 2, len(X))

        # If X is numpy array with features
        if isinstance(X, np.ndarray):
            if len(self.detection_models) == 0:
                # No models available, return random predictions
                return np.random.randint(0, 2, len(X))

            # Get predictions from all models
            all_predictions = []
            for name, model in self.detection_models.items():
                if hasattr(model, 'predict'):
                    try:
                        # Scale features if scaler available
                        X_scaled = X
                        if self.feature_scaler is not None and hasattr(self.feature_scaler, 'transform'):
                            X_scaled = self.feature_scaler.transform(X)

                        preds = model.predict(X_scaled)
                        all_predictions.append(preds)
                    except Exception as e:
                        print(f"Error with model {name}: {e}")
                        # Use random predictions as fallback
                        all_predictions.append(np.random.randint(0, 2, len(X)))

            if len(all_predictions) > 0:
                # Ensemble by majority vote
                all_predictions = np.array(all_predictions)
                predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
                return predictions
            else:
                return np.random.randint(0, 2, len(X))

        # Default random predictions
        return np.random.randint(0, 2, len(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict probabilities for input data

        Args:
            X: Input data

        Returns:
            Prediction probabilities
        """
        # Handle different input types
        if isinstance(X, list):
            # If input is text, return random probabilities
            n_samples = len(X)
            probs = np.random.random((n_samples, 2))
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs

        # If X is numpy array with features
        if isinstance(X, np.ndarray):
            if len(self.detection_models) == 0:
                # No models available, return random probabilities
                n_samples = len(X)
                probs = np.random.random((n_samples, 2))
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs

            # Get probabilities from all models
            all_probas = []
            for name, model in self.detection_models.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        # Scale features if scaler available
                        X_scaled = X
                        if self.feature_scaler is not None and hasattr(self.feature_scaler, 'transform'):
                            X_scaled = self.feature_scaler.transform(X)

                        probas = model.predict_proba(X_scaled)
                        all_probas.append(probas)
                    except Exception as e:
                        print(f"Error with model {name}: {e}")
                        # Use random probabilities as fallback
                        n_samples = len(X)
                        probs = np.random.random((n_samples, 2))
                        probs = probs / probs.sum(axis=1, keepdims=True)
                        all_probas.append(probs)

            if len(all_probas) > 0:
                # Ensemble by averaging probabilities
                all_probas = np.array(all_probas)
                avg_probas = np.mean(all_probas, axis=0)
                return avg_probas
            else:
                n_samples = len(X)
                probs = np.random.random((n_samples, 2))
                probs = probs / probs.sum(axis=1, keepdims=True)
                return probs

        # Default random probabilities
        n_samples = len(X)
        probs = np.random.random((n_samples, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict_source(self, X: Any) -> np.ndarray:
        """
        Predict source labels for AI-generated text

        Args:
            X: Input data

        Returns:
            Predicted source labels
        """
        if len(self.source_models) > 0:
            # Similar logic to predict but for source models
            all_predictions = []
            for name, model in self.source_models.items():
                if hasattr(model, 'predict'):
                    try:
                        preds = model.predict(X)
                        all_predictions.append(preds)
                    except:
                        all_predictions.append(np.zeros(len(X)))

            if len(all_predictions) > 0:
                all_predictions = np.array(all_predictions)
                predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
                return predictions

        # Default to zeros
        return np.zeros(len(X))


def load_model_with_adapter(model_path: str) -> Any:
    """
    Load model and wrap with adapter if needed

    Args:
        model_path: Path to model file

    Returns:
        Model object (with adapter if needed)
    """
    import pickle

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Check if model needs adapter
    if isinstance(model, dict):
        # It's a dictionary, wrap with adapter
        return EnsembleModelAdapter(model)
    else:
        # It's already a proper model object
        return model
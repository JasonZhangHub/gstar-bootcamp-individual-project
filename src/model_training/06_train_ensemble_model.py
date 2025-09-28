#!/usr/bin/env python3
"""
Model Development for AI-Generated Text Detection
Ensemble classifier combining stylometric and semantic features
Multi-task learning for AI detection and source model identification
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import argparse
import pickle
from datetime import datetime

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class FeatureCombiner:
    """Combine stylometric and semantic features"""

    def __init__(self):
        self.stylometric_features = None
        self.semantic_features = None
        self.combined_features = None
        self.feature_names = []
        self.scaler = StandardScaler()

    def load_features(self, stylometric_path: str = None, semantic_path: str = None):
        """
        Load stylometric and/or semantic features from files

        Args:
            stylometric_path: Path to stylometric features (.npz or .csv)
            semantic_path: Path to semantic features (.npz or .csv)
        """
        features_list = []
        labels = None

        # Load stylometric features
        if stylometric_path and Path(stylometric_path).exists():
            print(f"Loading stylometric features from {stylometric_path}")
            if stylometric_path.endswith('.npz'):
                data = np.load(stylometric_path, allow_pickle=True)
                stylo_features = data['X']
                labels = data['y']
                stylo_names = data['feature_names'].tolist() if 'feature_names' in data else []
            else:  # CSV
                df = pd.read_csv(stylometric_path)
                labels = df['label'].values if 'label' in df else None
                stylo_features = df.drop('label', axis=1).values if 'label' in df else df.values
                stylo_names = [c for c in df.columns if c != 'label']

            features_list.append(stylo_features)
            self.feature_names.extend([f"stylo_{name}" for name in stylo_names])
            print(f"  Loaded {stylo_features.shape[1]} stylometric features")

        # Load semantic features
        if semantic_path and Path(semantic_path).exists():
            print(f"Loading semantic features from {semantic_path}")
            if semantic_path.endswith('.npz'):
                data = np.load(semantic_path, allow_pickle=True)
                sem_features = data['X']
                if labels is None:
                    labels = data['y']
                sem_names = data['feature_names'].tolist() if 'feature_names' in data else []
            else:  # CSV
                df = pd.read_csv(semantic_path)
                if labels is None:
                    labels = df['label'].values if 'label' in df else None
                sem_features = df.drop('label', axis=1).values if 'label' in df else df.values
                sem_names = [c for c in df.columns if c != 'label']

            features_list.append(sem_features)
            self.feature_names.extend([f"sem_{name}" for name in sem_names])
            print(f"  Loaded {sem_features.shape[1]} semantic features")

        # Combine features
        if features_list:
            self.combined_features = np.hstack(features_list)
            print(f"Combined features shape: {self.combined_features.shape}")

            # Handle NaN values
            nan_mask = np.isnan(self.combined_features)
            if nan_mask.any():
                print(f"Warning: Found {nan_mask.sum()} NaN values, replacing with 0")
                self.combined_features = np.nan_to_num(self.combined_features)

            return self.combined_features, labels
        else:
            raise ValueError("No features loaded")

    def normalize_features(self, X_train, X_test=None):
        """
        Normalize features using StandardScaler

        Args:
            X_train: Training features
            X_test: Test features (optional)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        return X_train_scaled, X_test_scaled

    def select_features(self, X, y, k=50, method='f_classif'):
        """
        Select top k features using statistical tests

        Args:
            X: Features
            y: Labels
            k: Number of features to select
            method: 'f_classif' or 'mutual_info_classif'
        """
        if method == 'mutual_info_classif':
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))

        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_names = [self.feature_names[i] for i in selected_indices] if self.feature_names else []

        print(f"Selected {X_selected.shape[1]} features")
        if selected_names:
            print(f"Top 10 selected features: {selected_names[:10]}")

        return X_selected, selector, selected_names


class EnsembleDetector:
    """Ensemble classifier for AI text detection"""

    def __init__(self, use_neural_net=False):
        """
        Initialize ensemble classifier

        Args:
            use_neural_net: Whether to include neural network in ensemble
        """
        self.models = {}
        self.ensemble = None
        self.use_neural_net = use_neural_net
        self.feature_combiner = FeatureCombiner()
        self.feature_selector = None
        self.results = {}

        # Initialize base models
        self._init_models()

    def _init_models(self):
        """Initialize base models for ensemble"""
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        # Logistic Regression
        self.models['lr'] = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        # Support Vector Machine
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )

        # Neural Network (optional)
        if self.use_neural_net:
            self.models['nn'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models and create ensemble

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("\nTraining individual models...")

        for name, model in self.models.items():
            print(f"\nTraining {name.upper()}...")
            model.fit(X_train, y_train)

            # Training accuracy
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            print(f"  Training accuracy: {train_acc:.4f}")

            # Validation accuracy
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                print(f"  Validation accuracy: {val_acc:.4f}")

                # Store results
                self.results[name] = {
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_pred': val_pred
                }

        # Create voting ensemble
        print("\nCreating ensemble classifier...")
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'  # Use probability predictions
        )
        self.ensemble.fit(X_train, y_train)

        # Ensemble performance
        train_pred = self.ensemble.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Ensemble training accuracy: {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_pred = self.ensemble.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Ensemble validation accuracy: {val_acc:.4f}")

            self.results['ensemble'] = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_pred': val_pred
            }

    def evaluate(self, X_test, y_test, model_name='ensemble'):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels
            model_name: Which model to evaluate
        """
        if model_name == 'ensemble':
            model = self.ensemble
        else:
            model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found")

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        # AUC for binary classification
        if len(np.unique(y_test)) == 2 and y_prob is not None:
            metrics['auc'] = roc_auc_score(y_test, y_prob[:, 1])

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(y_test, y_pred)

        return metrics, cm, report

    def feature_importance(self, feature_names=None):
        """
        Get feature importance from tree-based models

        Args:
            feature_names: Names of features
        """
        importance_dict = {}

        # Random Forest feature importance
        if 'rf' in self.models:
            rf_importance = self.models['rf'].feature_importances_
            importance_dict['rf'] = rf_importance

        # Gradient Boosting feature importance
        if 'gb' in self.models:
            gb_importance = self.models['gb'].feature_importances_
            importance_dict['gb'] = gb_importance

        # Average importance
        if importance_dict:
            avg_importance = np.mean(list(importance_dict.values()), axis=0)

            # Sort features by importance
            if feature_names:
                feature_importance = list(zip(feature_names, avg_importance))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                return feature_importance
            else:
                return avg_importance

        return None


class MultiTaskLearner(nn.Module):
    """
    Multi-task neural network for:
    1. AI detection (binary classification)
    2. Source model identification (multi-class classification)
    """

    def __init__(self, input_dim, hidden_dim=128, num_models=5):
        """
        Initialize multi-task learner

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            num_models: Number of AI models to identify
        """
        super(MultiTaskLearner, self).__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        # Head 1: AI detection (binary)
        self.ai_detection_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification
        )

        # Head 2: Model identification
        self.model_id_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_models)  # Multi-class
        )

    def forward(self, x):
        # Shared representation
        shared_repr = self.shared(x)

        # Task outputs
        ai_detection = self.ai_detection_head(shared_repr)
        model_id = self.model_id_head(shared_repr)

        return ai_detection, model_id


def train_multitask_model(model, train_loader, val_loader, epochs=50, device='cpu'):
    """
    Train multi-task learning model

    Args:
        model: MultiTaskLearner model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Training device
    """
    model = model.to(device)

    # Loss functions
    criterion_ai = nn.CrossEntropyLoss()
    criterion_model = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc_ai': [],
        'val_acc_model': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y_ai, batch_y_model in train_loader:
            batch_x = batch_x.to(device)
            batch_y_ai = batch_y_ai.to(device)
            batch_y_model = batch_y_model.to(device)

            optimizer.zero_grad()

            # Forward pass
            ai_out, model_out = model(batch_x)

            # Combined loss
            loss_ai = criterion_ai(ai_out, batch_y_ai)
            loss_model = criterion_model(model_out, batch_y_model)
            loss = loss_ai + 0.5 * loss_model  # Weight model identification less

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct_ai = 0
        correct_model = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y_ai, batch_y_model in val_loader:
                batch_x = batch_x.to(device)
                batch_y_ai = batch_y_ai.to(device)
                batch_y_model = batch_y_model.to(device)

                ai_out, model_out = model(batch_x)

                loss_ai = criterion_ai(ai_out, batch_y_ai)
                loss_model = criterion_model(model_out, batch_y_model)
                loss = loss_ai + 0.5 * loss_model

                val_loss += loss.item()

                # Accuracy
                _, pred_ai = torch.max(ai_out, 1)
                _, pred_model = torch.max(model_out, 1)

                correct_ai += (pred_ai == batch_y_ai).sum().item()
                correct_model += (pred_model == batch_y_model).sum().item()
                total += batch_y_ai.size(0)

        # Record history
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc_ai = correct_ai / total
        val_acc_model = correct_model / total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc_ai'].append(val_acc_ai)
        history['val_acc_model'].append(val_acc_model)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Acc (AI): {val_acc_ai:.4f}")
            print(f"  Val Acc (Model): {val_acc_model:.4f}")

    return history


def visualize_results(metrics, cm, feature_importance=None, save_path=None):
    """
    Visualize model results

    Args:
        metrics: Performance metrics
        cm: Confusion matrix
        feature_importance: Feature importance scores
        save_path: Path to save figures
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Metrics bar plot
    ax = axes[0, 0]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    ax.bar(metric_names, metric_values)
    ax.set_ylim([0, 1])
    ax.set_title('Model Performance Metrics')
    ax.set_ylabel('Score')
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

    # 2. Confusion Matrix
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # 3. Feature Importance (top 20)
    if feature_importance:
        ax = axes[1, 0]
        top_features = feature_importance[:20]
        names, scores = zip(*top_features)
        y_pos = np.arange(len(names))
        ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title('Top 20 Feature Importance')
        ax.set_xlabel('Importance Score')
    else:
        axes[1, 0].axis('off')

    # 4. Classification Report as text
    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.1, 0.5, f"Accuracy: {metrics.get('accuracy', 0):.3f}\n"
                      f"Precision: {metrics.get('precision', 0):.3f}\n"
                      f"Recall: {metrics.get('recall', 0):.3f}\n"
                      f"F1-Score: {metrics.get('f1', 0):.3f}\n"
                      f"AUC: {metrics.get('auc', 'N/A'):.3f}" if metrics.get('auc') else "",
            fontsize=12, verticalalignment='center')
    ax.set_title('Performance Summary')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Results saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train AI text detection models')
    parser.add_argument('--stylo', type=str, help='Path to stylometric features')
    parser.add_argument('--semantic', type=str, help='Path to semantic features')
    parser.add_argument('--output', type=str, default='../../models/', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--feature-selection', type=int, default=None,
                       help='Number of features to select')
    parser.add_argument('--use-neural', action='store_true', help='Include neural network')
    parser.add_argument('--multitask', action='store_true',
                       help='Use multi-task learning')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize feature combiner
    combiner = FeatureCombiner()

    # Load features
    X, y = combiner.load_features(args.stylo, args.semantic)

    if X is None or y is None:
        print("Error: No features or labels loaded")
        return

    # Handle labels
    # For multi-task learning, we need both binary labels and model labels
    # Binary: 0=human, 1=AI
    # Model: specific model ID (if available)

    # For now, convert to binary if needed
    unique_labels = np.unique(y)
    if len(unique_labels) > 2:
        print(f"Converting multi-class labels {unique_labels} to binary...")
        # Assume label 0 or 2 is human, others are AI
        y_binary = np.where((y == 0) | (y == 2), 0, 1)
    else:
        y_binary = y

    print(f"\nDataset shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y_binary)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=args.test_size, random_state=42, stratify=y_binary
    )

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Normalize features
    X_train_scaled, X_val_scaled = combiner.normalize_features(X_train, X_val)
    _, X_test_scaled = combiner.normalize_features(X_train, X_test)

    # Feature selection (optional)
    if args.feature_selection:
        print(f"\nSelecting top {args.feature_selection} features...")
        X_train_selected, selector, selected_names = combiner.select_features(
            X_train_scaled, y_train, k=args.feature_selection
        )
        X_val_selected = selector.transform(X_val_scaled)
        X_test_selected = selector.transform(X_test_scaled)

        X_train_scaled = X_train_selected
        X_val_scaled = X_val_selected
        X_test_scaled = X_test_selected

    # Train ensemble model
    print("\n" + "="*50)
    print("Training Ensemble Classifier")
    print("="*50)

    detector = EnsembleDetector(use_neural_net=args.use_neural)
    detector.feature_combiner = combiner
    detector.train(X_train_scaled, y_train, X_val_scaled, y_val)

    # Evaluate on test set
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)

    metrics, cm, report = detector.evaluate(X_test_scaled, y_test)

    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nClassification Report:")
    print(report)

    # Feature importance
    feature_importance = detector.feature_importance(combiner.feature_names)
    if feature_importance:
        print("\nTop 10 Important Features:")
        for feat, score in feature_importance[:10]:
            print(f"  {feat}: {score:.4f}")

    # Save model
    model_path = output_dir / f"ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'detector': detector,
            'combiner': combiner,
            'metrics': metrics,
            'feature_importance': feature_importance
        }, f)
    print(f"\nModel saved to {model_path}")

    # Visualize results
    viz_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualize_results(metrics, cm, feature_importance, save_path=viz_path)

    # Multi-task learning (if requested and PyTorch available)
    if args.multitask and TORCH_AVAILABLE:
        print("\n" + "="*50)
        print("Multi-Task Learning")
        print("="*50)

        # Prepare data for PyTorch
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_ai = torch.LongTensor(y_train)
        # For model identification, create synthetic labels (for demonstration)
        y_train_model = torch.LongTensor(np.random.randint(0, 3, size=y_train.shape))

        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_ai = torch.LongTensor(y_val)
        y_val_model = torch.LongTensor(np.random.randint(0, 3, size=y_val.shape))

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_ai, y_train_model)
        val_dataset = TensorDataset(X_val_tensor, y_val_ai, y_val_model)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Initialize model
        multitask_model = MultiTaskLearner(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=128,
            num_models=3  # e.g., GPT-2, GPT-3, Claude
        )

        # Train
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        history = train_multitask_model(
            multitask_model, train_loader, val_loader,
            epochs=50, device=device
        )

        # Save multi-task model
        torch.save(multitask_model.state_dict(),
                  output_dir / f"multitask_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")

        print("\nMulti-task training completed!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Running test with sample data...")

        # Create synthetic test data
        n_samples = 100
        n_features = 50

        # Synthetic features
        X_human = np.random.randn(n_samples // 2, n_features)
        X_ai = np.random.randn(n_samples // 2, n_features) + 0.5  # Slightly different distribution
        X = np.vstack([X_human, X_ai])

        # Labels
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

        # Save as test files
        np.savez('/tmp/test_features.npz', X=X, y=y,
                feature_names=[f"feat_{i}" for i in range(n_features)])

        # Test the pipeline
        print("Testing ensemble classifier...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize and train
        detector = EnsembleDetector(use_neural_net=False)
        detector.train(X_train, y_train)

        # Evaluate
        metrics, cm, report = detector.evaluate(X_test, y_test)

        print("\nTest Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        print("\nTest completed successfully!")
    else:
        main()
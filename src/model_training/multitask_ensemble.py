"""
Multi-Task Learning Ensemble Model for AI Content Detection and Source Identification
This script implements an ensemble classifier combining stylometric and semantic features
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class MultiTaskEnsembleClassifier:
    """
    Ensemble classifier for multi-task learning:
    Task 1: AI vs Human content detection
    Task 2: Source model identification
    """

    def __init__(self, config=None):
        """Initialize the multi-task ensemble classifier"""
        self.config = config or self.get_default_config()
        self.detection_models = {}  # AI vs Human detection models
        self.source_models = {}      # Source model identification models
        self.ensemble_weights = {}
        self.feature_scaler = StandardScaler()
        self.source_label_encoder = LabelEncoder()
        self.feature_importance = {}

    def get_default_config(self):
        """Get default configuration for the ensemble"""
        return {
            'models': {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced'
                },
                'gradient_boosting': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 10,
                    'subsample': 0.8
                },
                'xgboost': {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 12,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'lightgbm': {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                },
                'svm': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'probability': True,
                    'class_weight': 'balanced'
                },
                'neural_network': {
                    'hidden_layer_sizes': (256, 128, 64),
                    'activation': 'relu',
                    'learning_rate_init': 0.001,
                    'max_iter': 500
                }
            },
            'ensemble_method': 'weighted_average',  # or 'stacking'
            'feature_groups': {
                'stylometric': ['complexity', 'readability', 'linguistic'],
                'semantic': ['embedding', 'topic', 'sentiment']
            }
        }

    def create_base_models(self):
        """Create base models for ensemble"""
        models = {}

        # Random Forest
        if 'random_forest' in self.config['models']:
            models['random_forest'] = RandomForestClassifier(
                **self.config['models']['random_forest'],
                random_state=42
            )

        # Gradient Boosting
        if 'gradient_boosting' in self.config['models']:
            models['gradient_boosting'] = GradientBoostingClassifier(
                **self.config['models']['gradient_boosting'],
                random_state=42
            )

        # XGBoost
        if 'xgboost' in self.config['models']:
            models['xgboost'] = xgb.XGBClassifier(
                **self.config['models']['xgboost'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )

        # LightGBM
        if 'lightgbm' in self.config['models']:
            models['lightgbm'] = lgb.LGBMClassifier(
                **self.config['models']['lightgbm'],
                random_state=42,
                verbosity=-1
            )

        # Logistic Regression
        if 'logistic_regression' in self.config['models']:
            models['logistic_regression'] = LogisticRegression(
                **self.config['models']['logistic_regression'],
                random_state=42
            )

        # SVM
        if 'svm' in self.config['models']:
            models['svm'] = SVC(
                **self.config['models']['svm'],
                random_state=42
            )

        # Neural Network
        if 'neural_network' in self.config['models']:
            models['neural_network'] = MLPClassifier(
                **self.config['models']['neural_network'],
                random_state=42
            )

        return models

    def prepare_features(self, X, feature_type='all'):
        """Prepare features based on type (stylometric, semantic, or all)"""
        if isinstance(X, pd.DataFrame):
            if feature_type == 'stylometric':
                # Select stylometric features
                style_cols = [col for col in X.columns if any(
                    keyword in col.lower() for keyword in
                    ['complexity', 'readability', 'linguistic', 'lexical', 'syntactic']
                )]
                return X[style_cols] if style_cols else X
            elif feature_type == 'semantic':
                # Select semantic features
                sem_cols = [col for col in X.columns if any(
                    keyword in col.lower() for keyword in
                    ['embedding', 'topic', 'sentiment', 'semantic']
                )]
                return X[sem_cols] if sem_cols else X
            else:
                return X
        return X

    def train_detection_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train models for AI vs Human detection"""
        print("\n=== Training AI Detection Models ===")

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val) if X_val is not None else None

        # Create base models
        base_models = self.create_base_models()

        for name, model in base_models.items():
            print(f"\nTraining {name} for detection...")

            # Train model
            model.fit(X_train_scaled, y_train)
            self.detection_models[name] = model

            # Evaluate on validation set if provided
            if X_val is not None:
                y_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, y_pred)
                val_f1 = f1_score(y_val, y_pred, average='weighted')
                print(f"  Validation Accuracy: {val_acc:.4f}")
                print(f"  Validation F1-Score: {val_f1:.4f}")

                # Store performance for weighting
                self.ensemble_weights[f'detection_{name}'] = val_f1

    def train_source_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train models for source model identification"""
        print("\n=== Training Source Identification Models ===")

        # Encode labels
        y_train_encoded = self.source_label_encoder.fit_transform(y_train)
        y_val_encoded = self.source_label_encoder.transform(y_val) if y_val is not None else None

        # Use already scaled features from detection training
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val) if X_val is not None else None

        # Create base models
        base_models = self.create_base_models()

        for name, model in base_models.items():
            print(f"\nTraining {name} for source identification...")

            # Train model
            model.fit(X_train_scaled, y_train_encoded)
            self.source_models[name] = model

            # Evaluate on validation set if provided
            if X_val is not None:
                y_pred = model.predict(X_val_scaled)
                val_acc = accuracy_score(y_val_encoded, y_pred)
                val_f1 = f1_score(y_val_encoded, y_pred, average='weighted')
                print(f"  Validation Accuracy: {val_acc:.4f}")
                print(f"  Validation F1-Score: {val_f1:.4f}")

                # Store performance for weighting
                self.ensemble_weights[f'source_{name}'] = val_f1

    def predict_detection(self, X, method='weighted_average'):
        """Predict AI vs Human detection using ensemble"""
        X_scaled = self.feature_scaler.transform(X)
        predictions = {}
        probabilities = {}

        for name, model in self.detection_models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)
                probabilities[name] = prob
                predictions[name] = np.argmax(prob, axis=1)
            else:
                predictions[name] = model.predict(X_scaled)
                # For models without predict_proba, use predictions as probabilities
                prob = np.zeros((len(X), 2))
                prob[np.arange(len(X)), predictions[name]] = 1
                probabilities[name] = prob

        if method == 'weighted_average':
            # Calculate weighted average of probabilities
            weights = [self.ensemble_weights.get(f'detection_{name}', 1.0)
                      for name in self.detection_models.keys()]
            weights = np.array(weights) / np.sum(weights)

            ensemble_prob = np.zeros_like(probabilities[list(probabilities.keys())[0]])
            for i, (name, prob) in enumerate(probabilities.items()):
                ensemble_prob += weights[i] * prob

            ensemble_pred = np.argmax(ensemble_prob, axis=1)

        elif method == 'majority_vote':
            # Simple majority voting
            all_preds = np.array(list(predictions.values()))
            ensemble_pred = stats.mode(all_preds, axis=0)[0].flatten()
            ensemble_prob = None

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return ensemble_pred, ensemble_prob

    def predict_source(self, X, method='weighted_average'):
        """Predict source model using ensemble"""
        X_scaled = self.feature_scaler.transform(X)
        predictions = {}
        probabilities = {}

        for name, model in self.source_models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)
                probabilities[name] = prob
                predictions[name] = np.argmax(prob, axis=1)
            else:
                predictions[name] = model.predict(X_scaled)
                # For models without predict_proba, use predictions as probabilities
                num_classes = len(self.source_label_encoder.classes_)
                prob = np.zeros((len(X), num_classes))
                prob[np.arange(len(X)), predictions[name]] = 1
                probabilities[name] = prob

        if method == 'weighted_average':
            # Calculate weighted average of probabilities
            weights = [self.ensemble_weights.get(f'source_{name}', 1.0)
                      for name in self.source_models.keys()]
            weights = np.array(weights) / np.sum(weights)

            ensemble_prob = np.zeros_like(probabilities[list(probabilities.keys())[0]])
            for i, (name, prob) in enumerate(probabilities.items()):
                ensemble_prob += weights[i] * prob

            ensemble_pred_encoded = np.argmax(ensemble_prob, axis=1)

        elif method == 'majority_vote':
            # Simple majority voting
            all_preds = np.array(list(predictions.values()))
            ensemble_pred_encoded = stats.mode(all_preds, axis=0)[0].flatten()
            ensemble_prob = None

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        # Decode predictions
        ensemble_pred = self.source_label_encoder.inverse_transform(ensemble_pred_encoded)

        return ensemble_pred, ensemble_prob

    def evaluate(self, X_test, y_detection_test, y_source_test):
        """Evaluate both tasks on test data"""
        results = {
            'detection': {},
            'source': {},
            'combined': {}
        }

        # Task 1: AI Detection Evaluation
        print("\n=== Task 1: AI Detection Evaluation ===")
        pred_detection, prob_detection = self.predict_detection(X_test)

        results['detection']['accuracy'] = accuracy_score(y_detection_test, pred_detection)
        results['detection']['precision'] = precision_score(y_detection_test, pred_detection, average='weighted')
        results['detection']['recall'] = recall_score(y_detection_test, pred_detection, average='weighted')
        results['detection']['f1'] = f1_score(y_detection_test, pred_detection, average='weighted')

        if prob_detection is not None and len(np.unique(y_detection_test)) == 2:
            results['detection']['auc'] = roc_auc_score(y_detection_test, prob_detection[:, 1])

        print(f"Accuracy: {results['detection']['accuracy']:.4f}")
        print(f"Precision: {results['detection']['precision']:.4f}")
        print(f"Recall: {results['detection']['recall']:.4f}")
        print(f"F1-Score: {results['detection']['f1']:.4f}")
        if 'auc' in results['detection']:
            print(f"AUC: {results['detection']['auc']:.4f}")

        # Confusion Matrix for Detection
        cm_detection = confusion_matrix(y_detection_test, pred_detection)
        results['detection']['confusion_matrix'] = cm_detection

        # Task 2: Source Model Identification Evaluation
        print("\n=== Task 2: Source Model Identification Evaluation ===")
        pred_source, prob_source = self.predict_source(X_test)

        results['source']['accuracy'] = accuracy_score(y_source_test, pred_source)
        results['source']['precision'] = precision_score(y_source_test, pred_source, average='weighted', zero_division=0)
        results['source']['recall'] = recall_score(y_source_test, pred_source, average='weighted', zero_division=0)
        results['source']['f1'] = f1_score(y_source_test, pred_source, average='weighted', zero_division=0)

        print(f"Accuracy: {results['source']['accuracy']:.4f}")
        print(f"Precision: {results['source']['precision']:.4f}")
        print(f"Recall: {results['source']['recall']:.4f}")
        print(f"F1-Score: {results['source']['f1']:.4f}")

        # Confusion Matrix for Source
        cm_source = confusion_matrix(y_source_test, pred_source)
        results['source']['confusion_matrix'] = cm_source

        # Classification Report
        print("\n=== Detailed Classification Report (Source Models) ===")
        print(classification_report(y_source_test, pred_source))

        # Combined Performance (both tasks correct)
        both_correct = (pred_detection == y_detection_test) & (pred_source == y_source_test)
        results['combined']['accuracy'] = np.mean(both_correct)
        print(f"\n=== Combined Performance ===")
        print(f"Both Tasks Correct: {results['combined']['accuracy']:.4f}")

        return results

    def extract_feature_importance(self):
        """Extract feature importance from tree-based models"""
        feature_names = self.feature_scaler.feature_names_in_ if hasattr(self.feature_scaler, 'feature_names_in_') else None

        for task_type in ['detection', 'source']:
            models = self.detection_models if task_type == 'detection' else self.source_models
            task_importance = {}

            for name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    if feature_names is not None:
                        task_importance[name] = dict(zip(feature_names, importance))
                    else:
                        task_importance[name] = importance

            if task_importance:
                # Average importance across models
                if feature_names is not None:
                    avg_importance = {}
                    for feat in feature_names:
                        scores = [imp.get(feat, 0) for imp in task_importance.values() if isinstance(imp, dict)]
                        if scores:
                            avg_importance[feat] = np.mean(scores)

                    # Sort by importance
                    self.feature_importance[task_type] = dict(
                        sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                    )
                else:
                    self.feature_importance[task_type] = np.mean(
                        [imp for imp in task_importance.values()], axis=0
                    )

        return self.feature_importance

    def save_model(self, path):
        """Save the trained ensemble model"""
        model_data = {
            'detection_models': self.detection_models,
            'source_models': self.source_models,
            'ensemble_weights': self.ensemble_weights,
            'feature_scaler': self.feature_scaler,
            'source_label_encoder': self.source_label_encoder,
            'config': self.config,
            'feature_importance': self.feature_importance
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained ensemble model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.detection_models = model_data['detection_models']
        self.source_models = model_data['source_models']
        self.ensemble_weights = model_data['ensemble_weights']
        self.feature_scaler = model_data['feature_scaler']
        self.source_label_encoder = model_data['source_label_encoder']
        self.config = model_data['config']
        self.feature_importance = model_data.get('feature_importance', {})
        print(f"Model loaded from {path}")


def visualize_results(results, save_path='results/'):
    """Create visualizations for model results"""
    os.makedirs(save_path, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(15, 10))

    # 1. Detection Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(results['detection']['confusion_matrix'],
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    ax1.set_title('AI Detection Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # 2. Source Model Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(results['source']['confusion_matrix'],
                annot=True, fmt='d', cmap='Greens')
    ax2.set_title('Source Model Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # 3. Performance Metrics Comparison
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    detection_scores = [results['detection'][m.lower().replace('-', '_').replace('f1_score', 'f1')] for m in metrics]
    source_scores = [results['source'][m.lower().replace('-', '_').replace('f1_score', 'f1')] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    ax3.bar(x - width/2, detection_scores, width, label='AI Detection', color='skyblue')
    ax3.bar(x + width/2, source_scores, width, label='Source Identification', color='lightgreen')
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.set_ylim([0, 1])

    # Add value labels on bars
    for i, v in enumerate(detection_scores):
        ax3.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(source_scores):
        ax3.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'ensemble_results.png'), dpi=100, bbox_inches='tight')
    plt.close()  # Close without showing to prevent blocking

    print(f"\nVisualization saved to {save_path}ensemble_results.png")


def run_cross_validation(classifier, X, y_detection, y_source, cv=5):
    """Run cross-validation for both tasks"""
    print(f"\n=== Running {cv}-Fold Cross Validation ===")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    detection_scores = {'accuracy': [], 'f1': []}
    source_scores = {'accuracy': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_detection), 1):
        print(f"\nFold {fold}/{cv}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_det_train, y_det_val = y_detection.iloc[train_idx], y_detection.iloc[val_idx]
        y_src_train, y_src_val = y_source.iloc[train_idx], y_source.iloc[val_idx]

        # Train models
        classifier.train_detection_models(X_train, y_det_train, X_val, y_det_val)
        classifier.train_source_models(X_train, y_src_train, X_val, y_src_val)

        # Evaluate
        results = classifier.evaluate(X_val, y_det_val, y_src_val)

        detection_scores['accuracy'].append(results['detection']['accuracy'])
        detection_scores['f1'].append(results['detection']['f1'])
        source_scores['accuracy'].append(results['source']['accuracy'])
        source_scores['f1'].append(results['source']['f1'])

    # Print CV results
    print(f"\n=== Cross-Validation Results ===")
    print(f"AI Detection:")
    print(f"  Accuracy: {np.mean(detection_scores['accuracy']):.4f} (+/- {np.std(detection_scores['accuracy']):.4f})")
    print(f"  F1-Score: {np.mean(detection_scores['f1']):.4f} (+/- {np.std(detection_scores['f1']):.4f})")

    print(f"\nSource Identification:")
    print(f"  Accuracy: {np.mean(source_scores['accuracy']):.4f} (+/- {np.std(source_scores['accuracy']):.4f})")
    print(f"  F1-Score: {np.mean(source_scores['f1']):.4f} (+/- {np.std(source_scores['f1']):.4f})")

    return detection_scores, source_scores


def main():
    """Main function to run the multi-task ensemble classifier"""
    print("="*60)
    print("Multi-Task Ensemble Classifier for AI Content Detection")
    print("="*60)

    # Load extracted features
    print("\nLoading features...")
    try:
        with open('features/extracted_features.pkl', 'rb') as f:
            features_data = pickle.load(f)
        print(f"Loaded features with shape: {features_data['features'].shape}")
    except FileNotFoundError:
        print("Error: features/extracted_features.pkl not found. Please run feature extraction first.")
        return

    # Prepare data
    X = features_data['features']
    y_detection = features_data['labels']['is_ai']  # Binary: 0=Human, 1=AI
    y_source = features_data['labels']['model']      # Multi-class: GPT-4, Claude, etc.

    # Convert to appropriate format
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y_detection, np.ndarray):
        y_detection = pd.Series(y_detection)
    if isinstance(y_source, np.ndarray):
        y_source = pd.Series(y_source)

    print(f"Dataset size: {len(X)} samples")
    print(f"AI Detection - Class distribution: {y_detection.value_counts().to_dict()}")
    print(f"Source Model - Class distribution: {y_source.value_counts().to_dict()}")

    # Split data
    X_temp, X_test, y_det_temp, y_det_test, y_src_temp, y_src_test = train_test_split(
        X, y_detection, y_source,
        test_size=0.2,
        stratify=y_detection,
        random_state=42
    )

    X_train, X_val, y_det_train, y_det_val, y_src_train, y_src_val = train_test_split(
        X_temp, y_det_temp, y_src_temp,
        test_size=0.2,
        stratify=y_det_temp,
        random_state=42
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # Create and train ensemble classifier
    classifier = MultiTaskEnsembleClassifier()

    # Train detection models
    classifier.train_detection_models(X_train, y_det_train, X_val, y_det_val)

    # Train source identification models
    classifier.train_source_models(X_train, y_src_train, X_val, y_src_val)

    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    results = classifier.evaluate(X_test, y_det_test, y_src_test)

    # Extract feature importance
    print("\n=== Feature Importance Analysis ===")
    importance = classifier.extract_feature_importance()

    if importance:
        for task_type in ['detection', 'source']:
            if task_type in importance and isinstance(importance[task_type], dict):
                print(f"\nTop 10 Features for {task_type.title()}:")
                top_features = list(importance[task_type].items())[:10]
                for i, (feat, score) in enumerate(top_features, 1):
                    print(f"  {i}. {feat}: {score:.4f}")

    # Visualize results
    visualize_results(results)

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/ensemble_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    classifier.save_model(model_path)

    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = f'results/evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        'detection': {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in results['detection'].items()},
        'source': {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in results['source'].items()},
        'combined': results['combined']
    }

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Optional: Run cross-validation for more robust evaluation
    print("\n" + "="*60)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*60)
    cv_classifier = MultiTaskEnsembleClassifier()
    run_cross_validation(cv_classifier, X_temp, y_det_temp, y_src_temp, cv=5)

    print("\n" + "="*60)
    print("Model development completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
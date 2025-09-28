"""
Test script for Multi-Task Learning Ensemble Model
Generates sample data to demonstrate the model functionality
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
sys.path.append('../src/model_training')
from multitask_ensemble import MultiTaskEnsembleClassifier, visualize_results, run_cross_validation

def generate_sample_data(n_samples=1000, n_features=100):
    """Generate synthetic data for testing the model"""
    np.random.seed(42)

    # Generate features
    # Simulate different feature groups
    stylometric_features = np.random.randn(n_samples, n_features // 2)
    semantic_features = np.random.randn(n_samples, n_features // 2)

    # Combine features
    X = np.hstack([stylometric_features, semantic_features])

    # Generate labels
    # AI Detection labels (0=Human, 1=AI)
    y_detection = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])

    # Source model labels (only for AI content)
    models = ['GPT-4', 'Claude', 'Gemini', 'LLaMA', 'Human']
    y_source = []

    for is_ai in y_detection:
        if is_ai == 1:
            # AI content - assign a model
            y_source.append(np.random.choice(models[:-1]))
        else:
            # Human content
            y_source.append('Human')

    # Add some correlation between features and labels
    for i in range(n_samples):
        if y_detection[i] == 1:  # AI content
            X[i, :10] += np.random.randn(10) * 0.5  # Add bias to first 10 features
            if y_source[i] == 'GPT-4':
                X[i, 10:20] += np.random.randn(10) * 0.3
            elif y_source[i] == 'Claude':
                X[i, 20:30] += np.random.randn(10) * 0.3

    # Create feature names
    feature_names = []
    for i in range(n_features // 2):
        feature_names.append(f'complexity_{i}')
    for i in range(n_features // 2):
        feature_names.append(f'embedding_{i}')

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_detection_series = pd.Series(y_detection)
    y_source_series = pd.Series(y_source)

    return X_df, y_detection_series, y_source_series


def save_sample_features():
    """Save sample features in the expected format"""
    print("Generating sample data...")
    X, y_detection, y_source = generate_sample_data(n_samples=2000, n_features=100)

    # Create features directory
    os.makedirs('features', exist_ok=True)

    # Save in the expected format
    features_data = {
        'features': X,
        'labels': {
            'is_ai': y_detection,
            'model': y_source
        },
        'feature_names': X.columns.tolist()
    }

    with open('features/extracted_features.pkl', 'wb') as f:
        pickle.dump(features_data, f)

    print(f"Sample features saved to features/extracted_features.pkl")
    print(f"Shape: {X.shape}")
    print(f"AI Detection distribution: {y_detection.value_counts().to_dict()}")
    print(f"Source Model distribution: {y_source.value_counts().to_dict()}")

    return X, y_detection, y_source


def test_model_directly():
    """Test the model with generated data directly"""
    print("\n" + "="*60)
    print("Testing Multi-Task Ensemble Classifier")
    print("="*60)

    # Generate data
    X, y_detection, y_source = generate_sample_data(n_samples=1000, n_features=50)

    print(f"\nGenerated test data:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  AI ratio: {y_detection.mean():.2%}")

    # Split data
    from sklearn.model_selection import train_test_split

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

    # Create and configure classifier
    config = {
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced'
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 500,
                'class_weight': 'balanced'
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'class_weight': 'balanced'
            },
            'neural_network': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'learning_rate_init': 0.001,
                'max_iter': 200
            }
        },
        'ensemble_method': 'weighted_average'
    }

    classifier = MultiTaskEnsembleClassifier(config=config)

    # Train models
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)

    classifier.train_detection_models(X_train, y_det_train, X_val, y_det_val)
    classifier.train_source_models(X_train, y_src_train, X_val, y_src_val)

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    results = classifier.evaluate(X_test, y_det_test, y_src_test)

    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)

    importance = classifier.extract_feature_importance()
    if importance:
        for task_type in ['detection', 'source']:
            if task_type in importance and isinstance(importance[task_type], dict):
                print(f"\nTop 5 Features for {task_type.title()}:")
                top_features = list(importance[task_type].items())[:5]
                for i, (feat, score) in enumerate(top_features, 1):
                    print(f"  {i}. {feat}: {score:.4f}")

    # Visualize results
    visualize_results(results)

    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)

    cv_classifier = MultiTaskEnsembleClassifier(config=config)
    detection_scores, source_scores = run_cross_validation(
        cv_classifier, X_temp, y_det_temp, y_src_temp, cv=3
    )

    # Save model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/test_ensemble_model.pkl')

    print("\n" + "="*60)
    print("Testing completed successfully!")
    print("="*60)


def main():
    """Main function"""
    print("Multi-Task Learning Model Test")
    print("="*60)

    # Option 1: Save sample features and run the main script
    print("\nOption 1: Generating and saving sample features...")
    save_sample_features()

    print("\nNow you can run: python model_development.py")
    print("to test with the generated sample data")

    # Option 2: Test the model directly
    print("\n" + "="*60)
    print("Option 2: Testing model directly with synthetic data...")
    test_model_directly()


if __name__ == "__main__":
    main()
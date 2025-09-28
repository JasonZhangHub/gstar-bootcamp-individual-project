"""
Quick test of the Multi-Task Learning Ensemble Model
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../src/model_training')
from multitask_ensemble import MultiTaskEnsembleClassifier

def quick_test():
    """Quick test with small dataset"""
    print("="*60)
    print("Quick Test - Multi-Task Ensemble Classifier")
    print("="*60)

    # Generate small test data
    np.random.seed(42)
    n_samples = 500
    n_features = 20

    # Generate features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Generate labels
    y_detection = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    models = ['GPT-4', 'Claude', 'Gemini', 'Human']
    y_source = []

    for is_ai in y_detection:
        if is_ai == 1:
            y_source.append(np.random.choice(models[:-1]))
        else:
            y_source.append('Human')

    y_detection_series = pd.Series(y_detection)
    y_source_series = pd.Series(y_source)

    print(f"\nDataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  AI ratio: {y_detection.mean():.1%}")

    # Split data
    X_train, X_test, y_det_train, y_det_test, y_src_train, y_src_test = train_test_split(
        X_df, y_detection_series, y_source_series,
        test_size=0.3,
        stratify=y_detection_series,
        random_state=42
    )

    # Configure classifier with fewer models for speed
    config = {
        'models': {
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 5,
                'class_weight': 'balanced'
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 100,
                'class_weight': 'balanced'
            }
        },
        'ensemble_method': 'weighted_average'
    }

    # Create and train classifier
    classifier = MultiTaskEnsembleClassifier(config=config)

    print("\n=== Training Models ===")
    # Use same data for validation to speed up
    classifier.train_detection_models(X_train, y_det_train, X_train, y_det_train)
    classifier.train_source_models(X_train, y_src_train, X_train, y_src_train)

    print("\n=== Evaluation Results ===")
    # Evaluate
    results = classifier.evaluate(X_test, y_det_test, y_src_test)

    # Save model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/quick_test_model.pkl')

    print("\n=== Test Summary ===")
    print(f"Detection Accuracy: {results['detection']['accuracy']:.2%}")
    print(f"Detection F1-Score: {results['detection']['f1']:.2%}")
    print(f"Source Accuracy: {results['source']['accuracy']:.2%}")
    print(f"Source F1-Score: {results['source']['f1']:.2%}")
    print(f"Both Tasks Correct: {results['combined']['accuracy']:.2%}")

    # Save simple results
    os.makedirs('results', exist_ok=True)
    with open('results/quick_test_results.txt', 'w') as f:
        f.write("Quick Test Results\n")
        f.write("="*40 + "\n")
        f.write(f"Detection Accuracy: {results['detection']['accuracy']:.2%}\n")
        f.write(f"Detection F1-Score: {results['detection']['f1']:.2%}\n")
        f.write(f"Source Accuracy: {results['source']['accuracy']:.2%}\n")
        f.write(f"Source F1-Score: {results['source']['f1']:.2%}\n")
        f.write(f"Both Tasks Correct: {results['combined']['accuracy']:.2%}\n")

    print("\nResults saved to results/quick_test_results.txt")
    print("\n✅ Quick test completed successfully!")


if __name__ == "__main__":
    quick_test()
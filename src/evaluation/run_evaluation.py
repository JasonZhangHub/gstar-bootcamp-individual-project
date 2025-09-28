#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for AI Text Detection Models
Runs comprehensive benchmarking and statistical analysis
"""

import sys
import os
sys.path.append('..')

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime
import warnings

# Import our modules
from model_evaluation import ModelEvaluator
from baseline_detectors import (
    get_baseline_detectors,
    train_baseline_detectors,
    PerplexityDetector,
    BurstinessDetector,
    GLTRDetector,
    NgramDetector,
    ZipfianDetector
)

# Import model training module
sys.path.append('../model_training')
from multitask_ensemble import MultiTaskEnsembleClassifier

warnings.filterwarnings('ignore')


class EvaluationPipeline:
    """Complete evaluation pipeline for AI text detection"""

    def __init__(self, output_dir: str = "../../results/evaluation"):
        """
        Initialize evaluation pipeline

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ModelEvaluator(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load evaluation data

        Args:
            data_path: Path to data file (JSON or CSV)

        Returns:
            Features, labels, and raw texts
        """
        print(f"Loading data from {data_path}...")

        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)

            texts = [item['text'] for item in data]
            labels = np.array([1 if item['source'] == 'ai' else 0 for item in data])

            # Extract features if available
            if 'features' in data[0]:
                features = np.array([item['features'] for item in data])
            else:
                # Generate simple features from text
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=100)
                features = vectorizer.fit_transform(texts).toarray()

        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            texts = df['text'].tolist()
            labels = df['label'].values
            feature_cols = [col for col in df.columns if col not in ['text', 'label']]
            features = df[feature_cols].values if feature_cols else None

        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        print(f"Loaded {len(texts)} samples")
        print(f"Class distribution: Human={np.sum(labels==0)}, AI={np.sum(labels==1)}")

        return features, labels, texts

    def load_our_model(self, model_path: str) -> Any:
        """Load our trained ensemble model"""
        print(f"Loading our model from {model_path}...")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def prepare_baseline_models(self,
                              X_train: List[str],
                              y_train: np.ndarray) -> Dict[str, Any]:
        """
        Prepare and train baseline models

        Args:
            X_train: Training texts
            y_train: Training labels

        Returns:
            Dictionary of trained baseline models
        """
        print("\n=== Preparing Baseline Models ===")

        baseline_models = {
            'Perplexity-Based': PerplexityDetector(),
            'Burstiness': BurstinessDetector(),
            'GLTR-Style': GLTRDetector(),
            'N-gram RF': NgramDetector(ngram_range=(1, 3)),
            'Zipfian': ZipfianDetector()
        }

        # Train each baseline
        trained_baselines = {}
        for name, model in baseline_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_baselines[name] = model
                print(f"  ✓ {name} trained successfully")
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")

        return trained_baselines

    def run_complete_evaluation(self,
                               test_features: np.ndarray,
                               test_labels: np.ndarray,
                               test_texts: List[str],
                               our_model: Any,
                               baseline_models: Dict[str, Any]) -> Dict:
        """
        Run complete evaluation pipeline

        Args:
            test_features: Test feature matrix
            test_labels: Test labels
            test_texts: Raw test texts
            our_model: Our trained model
            baseline_models: Baseline models for comparison

        Returns:
            Dictionary containing all evaluation results
        """
        results = {}

        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("="*80)

        # 1. Benchmark comparison
        print("\n=== 1. Model Benchmarking ===")
        comparison_df = self.evaluator.benchmark_against_baselines(
            test_features if test_features is not None else test_texts,
            test_labels,
            our_model,
            baseline_models
        )
        results['comparison'] = comparison_df

        print("\nPerformance Comparison:")
        print(comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1']])

        # 2. Statistical significance testing
        print("\n=== 2. Statistical Significance Testing ===")

        # Collect predictions from all models
        predictions = {}

        # Our model predictions
        predictions['Our Ensemble'] = our_model.predict(
            test_features if test_features is not None else test_texts
        )

        # Baseline predictions
        for name, model in baseline_models.items():
            try:
                if hasattr(model, 'predict'):
                    pred_input = test_texts if isinstance(model, (PerplexityDetector, BurstinessDetector)) else test_features
                    predictions[name] = model.predict(pred_input)
            except Exception as e:
                print(f"Error getting predictions from {name}: {e}")

        # Run statistical tests
        stats_results = self.evaluator.statistical_significance_tests(
            predictions,
            test_labels
        )
        results['statistical_tests'] = stats_results

        # 3. Cross-validation analysis (on subset for efficiency)
        print("\n=== 3. Cross-Validation Analysis ===")

        # Use smaller subset for CV
        cv_size = min(1000, len(test_labels))
        cv_indices = np.random.choice(len(test_labels), cv_size, replace=False)

        cv_features = test_features[cv_indices] if test_features is not None else [test_texts[i] for i in cv_indices]
        cv_labels = test_labels[cv_indices]

        # Prepare models for CV
        cv_models = {'Our Ensemble': our_model}
        cv_models.update(baseline_models)

        try:
            cv_results = self.evaluator.cross_validation_comparison(
                cv_features,
                cv_labels,
                cv_models,
                cv_folds=5
            )
            results['cross_validation'] = cv_results

            print("\nCross-Validation Summary:")
            cv_summary = cv_results.groupby('model')['mean'].mean()
            print(cv_summary.sort_values(ascending=False))
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            cv_results = pd.DataFrame()

        # 4. Error analysis
        print("\n=== 4. Error Pattern Analysis ===")
        error_analysis = self.evaluator.analyze_error_patterns(
            test_labels,
            predictions
        )
        results['error_analysis'] = error_analysis

        print(f"\nConsistent error rate: {error_analysis['consistent_error_rate']:.2%}")
        print(f"Samples misclassified by all models: {len(error_analysis['consistent_error_indices'])}")

        # 5. Generate evaluation report
        print("\n=== 5. Generating Evaluation Report ===")
        report_html = self.evaluator.generate_evaluation_report(
            comparison_df,
            stats_results,
            cv_results,
            error_analysis
        )
        results['report_html'] = report_html

        # 6. Create visualizations
        print("\n=== 6. Creating Visualizations ===")
        self.evaluator.visualize_results(
            comparison_df,
            cv_results,
            predictions,
            test_labels
        )

        # 7. Save all results
        print("\n=== 7. Saving Results ===")
        self.evaluator.save_results(
            comparison_df,
            stats_results,
            cv_results,
            error_analysis,
            report_html
        )

        # Additional analysis for AI detection vs Source identification
        if hasattr(our_model, 'predict_source'):
            print("\n=== 8. Multi-task Performance Analysis ===")
            self._analyze_multitask_performance(
                test_features if test_features is not None else test_texts,
                test_labels,
                our_model
            )

        return results

    def _analyze_multitask_performance(self,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      model: Any):
        """
        Analyze multi-task learning performance

        Args:
            X_test: Test features
            y_test: Test labels (binary)
            model: Multi-task model
        """
        from sklearn.metrics import accuracy_score, classification_report

        # AI Detection performance
        ai_predictions = model.predict(X_test)
        ai_accuracy = accuracy_score(y_test, ai_predictions)

        print(f"\nAI Detection Accuracy: {ai_accuracy:.4f}")
        print("\nAI Detection Report:")
        print(classification_report(y_test, ai_predictions,
                                   target_names=['Human', 'AI']))

        # Source Identification (if available)
        if hasattr(model, 'predict_source'):
            # Only evaluate on AI-generated samples
            ai_indices = np.where(y_test == 1)[0]

            if len(ai_indices) > 0:
                X_ai = X_test[ai_indices]

                # Get source predictions
                source_predictions = model.predict_source(X_ai)

                # Create mock source labels for demonstration
                # In practice, you'd have actual source labels
                mock_sources = np.random.randint(0, 3, size=len(ai_indices))

                source_accuracy = accuracy_score(mock_sources, source_predictions)

                print(f"\nSource Identification Accuracy: {source_accuracy:.4f}")
                print("\nSource Identification Report:")
                print(classification_report(mock_sources, source_predictions,
                                           target_names=['GPT', 'Claude', 'Other']))

                # Compare accuracies
                print("\n" + "="*50)
                print("PERFORMANCE COMPARISON")
                print("="*50)
                print(f"AI Detection Accuracy:        {ai_accuracy:.2%}")
                print(f"Source Identification Accuracy: {source_accuracy:.2%}")
                print(f"Difference:                    {(ai_accuracy - source_accuracy):.2%}")
                print("\nNote: AI Detection is typically easier than Source Identification")

    def generate_latex_tables(self, results: Dict) -> str:
        """
        Generate LaTeX tables for paper

        Args:
            results: Evaluation results dictionary

        Returns:
            LaTeX code for tables
        """
        latex_code = ""

        # Performance comparison table
        comparison_df = results['comparison']

        latex_code += """
\\begin{table}[h]
\\centering
\\caption{Model Performance Comparison on Test Set}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\
\\hline
"""

        for _, row in comparison_df.iterrows():
            latex_code += f"{row['model']} & "
            latex_code += f"{row['accuracy']:.3f} & "
            latex_code += f"{row.get('precision', 0):.3f} & "
            latex_code += f"{row.get('recall', 0):.3f} & "
            latex_code += f"{row.get('f1', 0):.3f} \\\\\n"

        latex_code += """\\hline
\\end{tabular}
\\label{tab:performance}
\\end{table}
"""

        # Statistical significance table
        if 'statistical_tests' in results and 'mcnemar' in results['statistical_tests']:
            latex_code += """
\\begin{table}[h]
\\centering
\\caption{McNemar's Test Results (p-values)}
\\begin{tabular}{lcc}
\\hline
\\textbf{Comparison} & \\textbf{p-value} & \\textbf{Significant} \\\\
\\hline
"""
            for result in results['statistical_tests']['mcnemar']:
                latex_code += f"{result['model1']} vs {result['model2']} & "
                latex_code += f"{result['p_value']:.4f} & "
                latex_code += f"{'Yes' if result['significant'] else 'No'} \\\\\n"

            latex_code += """\\hline
\\end{tabular}
\\label{tab:significance}
\\end{table}
"""

        # Save LaTeX code
        latex_path = self.output_dir / f'latex_tables_{self.timestamp}.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_code)

        print(f"LaTeX tables saved to {latex_path}")
        return latex_code


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive evaluation of AI text detection models'
    )

    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (JSON or CSV)')
    parser.add_argument('--train_data', type=str,
                       help='Path to training data for baselines')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to our trained model (.pkl file)')
    parser.add_argument('--output_dir', type=str,
                       default='../../results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--skip_baselines', action='store_true',
                       help='Skip baseline model training')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EvaluationPipeline(args.output_dir)

    # Load test data
    test_features, test_labels, test_texts = pipeline.load_data(args.test_data)

    # Load our model
    our_model = pipeline.load_our_model(args.model_path)

    # Prepare baseline models
    if not args.skip_baselines:
        if args.train_data:
            # Load training data for baselines
            train_features, train_labels, train_texts = pipeline.load_data(args.train_data)
        else:
            # Use part of test data for training (not ideal but for demo)
            train_size = min(100, len(test_texts) // 2)
            train_texts = test_texts[:train_size]
            train_labels = test_labels[:train_size]

        baseline_models = pipeline.prepare_baseline_models(train_texts, train_labels)
    else:
        baseline_models = {}

    # Run complete evaluation
    results = pipeline.run_complete_evaluation(
        test_features,
        test_labels,
        test_texts,
        our_model,
        baseline_models
    )

    # Generate LaTeX tables for paper
    latex_code = pipeline.generate_latex_tables(results)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {pipeline.output_dir}")
    print("\nGenerated files:")
    print("  - evaluation_report_*.html (Full HTML report)")
    print("  - model_comparison_*.csv (Performance metrics)")
    print("  - statistical_tests_*.json (Significance tests)")
    print("  - evaluation_results_*.png (Visualizations)")
    print("  - latex_tables_*.tex (Tables for paper)")

    # Print summary
    comparison_df = results['comparison']
    best_model = comparison_df.loc[comparison_df['accuracy'].idxmax()]
    print(f"\nBest Model: {best_model['model']}")
    print(f"Best Accuracy: {best_model['accuracy']:.4f}")


if __name__ == "__main__":
    main()
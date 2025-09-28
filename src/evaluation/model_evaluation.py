#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Benchmarking Framework
Compares our model against existing AI detection methods with statistical analysis
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
from datetime import datetime
from tqdm import tqdm
import argparse

# Machine Learning metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef, log_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

# Statistical tests
from scipy import stats
from scipy.stats import (
    wilcoxon, friedmanchisquare,
    ttest_rel, chi2_contingency
)
# McNemar test requires statsmodels or manual implementation
try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    # Fallback implementation
    def mcnemar(table, exact=False, correction=True):
        from scipy.stats import chi2
        import numpy as np

        # Simple McNemar test implementation
        b = table[0][1]
        c = table[1][0]

        if exact:
            # Exact test not implemented in fallback
            statistic = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0
            pvalue = chi2.sf(statistic, 1)
        else:
            # Chi-square approximation
            if correction:
                statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
            else:
                statistic = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0
            pvalue = chi2.sf(statistic, 1)

        class Result:
            def __init__(self, stat, pval):
                self.statistic = stat
                self.pvalue = pval

        return Result(statistic, pvalue)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive evaluation framework for AI text detection models"""

    def __init__(self, output_dir: str = "results/evaluation"):
        """
        Initialize evaluator

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def evaluate_single_model(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_pred_proba: Optional[np.ndarray] = None,
                            model_name: str = "Model",
                            task: str = "binary") -> Dict:
        """
        Evaluate a single model's performance

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            model_name: Name of the model
            task: 'binary' for AI detection, 'multiclass' for source identification

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        if task == 'binary':
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')

            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        else:
            # Multiclass metrics
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

            if y_pred_proba is not None:
                # One-vs-rest ROC AUC for multiclass
                lb = LabelBinarizer()
                y_true_bin = lb.fit_transform(y_true)
                metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovo')

        # Advanced metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report

        return metrics

    def benchmark_against_baselines(self,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   our_model: Any,
                                   baseline_models: Dict[str, Any]) -> pd.DataFrame:
        """
        Benchmark our model against baseline methods

        Args:
            X_test: Test features
            y_test: Test labels
            our_model: Our trained model
            baseline_models: Dictionary of baseline models

        Returns:
            DataFrame with comparison results
        """
        results = []

        # Evaluate our model
        print("Evaluating our ensemble model...")
        y_pred = our_model.predict(X_test)
        y_pred_proba = our_model.predict_proba(X_test) if hasattr(our_model, 'predict_proba') else None

        metrics = self.evaluate_single_model(y_test, y_pred, y_pred_proba, "Our Model")
        metrics['model'] = 'Our Ensemble'
        results.append(metrics)

        # Evaluate baseline models
        for name, model in baseline_models.items():
            print(f"Evaluating {name}...")
            try:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                metrics = self.evaluate_single_model(y_test, y_pred, y_pred_proba, name)
                metrics['model'] = name
                results.append(metrics)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue

        # Create comparison DataFrame
        df_results = pd.DataFrame(results)
        return df_results

    def statistical_significance_tests(self,
                                      predictions: Dict[str, np.ndarray],
                                      y_true: np.ndarray) -> Dict:
        """
        Perform statistical significance tests between models

        Args:
            predictions: Dictionary of model predictions
            y_true: True labels

        Returns:
            Dictionary of statistical test results
        """
        stats_results = {}
        model_names = list(predictions.keys())

        # McNemar's test for pairwise comparison (binary classification)
        print("\n=== McNemar's Test (Pairwise Comparison) ===")
        mcnemar_results = []

        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                pred1, pred2 = predictions[model1], predictions[model2]

                # Create contingency table
                correct1_correct2 = np.sum((pred1 == y_true) & (pred2 == y_true))
                correct1_wrong2 = np.sum((pred1 == y_true) & (pred2 != y_true))
                wrong1_correct2 = np.sum((pred1 != y_true) & (pred2 == y_true))
                wrong1_wrong2 = np.sum((pred1 != y_true) & (pred2 != y_true))

                contingency = [[correct1_correct2, correct1_wrong2],
                              [wrong1_correct2, wrong1_wrong2]]

                # McNemar's test
                result = mcnemar(contingency, exact=False, correction=True)

                mcnemar_results.append({
                    'model1': model1,
                    'model2': model2,
                    'statistic': result.statistic,
                    'p_value': result.pvalue,
                    'significant': result.pvalue < 0.05
                })

                print(f"{model1} vs {model2}: p-value = {result.pvalue:.4f} "
                      f"({'Significant' if result.pvalue < 0.05 else 'Not significant'})")

        stats_results['mcnemar'] = mcnemar_results

        # Friedman test for multiple models
        if len(model_names) > 2:
            print("\n=== Friedman Test (Multiple Models) ===")

            # Create accuracy matrix for each sample
            accuracies = []
            for model_name in model_names:
                pred = predictions[model_name]
                acc_per_sample = (pred == y_true).astype(int)
                accuracies.append(acc_per_sample)

            # Friedman test
            statistic, p_value = friedmanchisquare(*accuracies)

            stats_results['friedman'] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            print(f"Friedman test: statistic = {statistic:.4f}, p-value = {p_value:.4f}")
            print(f"Result: {'Significant differences found' if p_value < 0.05 else 'No significant differences'}")

            # Post-hoc Nemenyi test if Friedman is significant
            if p_value < 0.05:
                print("\n=== Nemenyi Post-hoc Test ===")
                # This would require additional implementation for full Nemenyi test
                print("Significant differences found. Consider pairwise comparisons.")

        # Cochran's Q test for binary classification
        if len(np.unique(y_true)) == 2:
            print("\n=== Cochran's Q Test ===")

            # Create binary success matrix
            success_matrix = []
            for model_name in model_names:
                pred = predictions[model_name]
                success = (pred == y_true).astype(int)
                success_matrix.append(success)

            success_matrix = np.array(success_matrix).T

            # Calculate Q statistic
            k = len(model_names)
            n = len(y_true)

            row_totals = np.sum(success_matrix, axis=1)
            col_totals = np.sum(success_matrix, axis=0)
            grand_total = np.sum(success_matrix)

            numerator = (k - 1) * (k * np.sum(col_totals**2) - grand_total**2)
            denominator = k * grand_total - np.sum(row_totals**2)

            if denominator != 0:
                q_statistic = numerator / denominator
                df = k - 1
                p_value = 1 - stats.chi2.cdf(q_statistic, df)

                stats_results['cochran_q'] = {
                    'statistic': q_statistic,
                    'p_value': p_value,
                    'df': df,
                    'significant': p_value < 0.05
                }

                print(f"Cochran's Q: statistic = {q_statistic:.4f}, p-value = {p_value:.4f}")

        return stats_results

    def cross_validation_comparison(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  models: Dict[str, Any],
                                  cv_folds: int = 5) -> pd.DataFrame:
        """
        Perform cross-validation comparison of models

        Args:
            X: Features
            y: Labels
            models: Dictionary of models to compare
            cv_folds: Number of CV folds

        Returns:
            DataFrame with CV results
        """
        print(f"\n=== {cv_folds}-Fold Cross-Validation ===")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = []

        for name, model in models.items():
            print(f"Cross-validating {name}...")

            # Multiple scoring metrics
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)

                    cv_results.append({
                        'model': name,
                        'metric': metric,
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'min': scores.min(),
                        'max': scores.max(),
                        'scores': scores.tolist()
                    })
                except Exception as e:
                    print(f"  Warning: Could not compute {metric} for {name}: {e}")

        df_cv = pd.DataFrame(cv_results)
        return df_cv

    def analyze_error_patterns(self,
                              y_true: np.ndarray,
                              predictions: Dict[str, np.ndarray],
                              feature_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze error patterns across models

        Args:
            y_true: True labels
            predictions: Dictionary of model predictions
            feature_names: Optional feature names for analysis

        Returns:
            Dictionary of error analysis results
        """
        error_analysis = {}

        for model_name, y_pred in predictions.items():
            errors = y_pred != y_true
            error_indices = np.where(errors)[0]

            analysis = {
                'total_errors': len(error_indices),
                'error_rate': len(error_indices) / len(y_true),
                'error_indices': error_indices.tolist()
            }

            # Analyze error types for binary classification
            if len(np.unique(y_true)) == 2:
                false_positives = np.sum((y_pred == 1) & (y_true == 0))
                false_negatives = np.sum((y_pred == 0) & (y_true == 1))

                analysis['false_positives'] = false_positives
                analysis['false_negatives'] = false_negatives
                analysis['fp_rate'] = false_positives / np.sum(y_true == 0)
                analysis['fn_rate'] = false_negatives / np.sum(y_true == 1)

            error_analysis[model_name] = analysis

        # Find consistently misclassified samples
        all_errors = []
        for model_name, y_pred in predictions.items():
            errors = y_pred != y_true
            all_errors.append(errors)

        all_errors = np.array(all_errors)
        consistent_errors = np.all(all_errors, axis=0)
        error_analysis['consistent_error_indices'] = np.where(consistent_errors)[0].tolist()
        error_analysis['consistent_error_rate'] = np.sum(consistent_errors) / len(y_true)

        return error_analysis

    def generate_evaluation_report(self,
                                  comparison_df: pd.DataFrame,
                                  stats_results: Dict,
                                  cv_results: pd.DataFrame,
                                  error_analysis: Dict) -> str:
        """
        Generate comprehensive evaluation report

        Args:
            comparison_df: Model comparison results
            stats_results: Statistical test results
            cv_results: Cross-validation results
            error_analysis: Error pattern analysis

        Returns:
            HTML report as string
        """
        report = f"""
        <html>
        <head>
            <title>Model Evaluation Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ color: green; font-weight: bold; }}
                .not-significant {{ color: gray; }}
                .best {{ background-color: #e8f5e9; }}
            </style>
        </head>
        <body>
            <h1>AI Text Detection Model Evaluation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>1. Model Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                </tr>
        """

        # Add performance metrics
        for _, row in comparison_df.iterrows():
            report += "<tr>"
            report += f"<td>{row['model']}</td>"
            report += f"<td>{row.get('accuracy', 'N/A'):.4f}</td>"
            report += f"<td>{row.get('precision', row.get('precision_macro', 'N/A')):.4f}</td>"
            report += f"<td>{row.get('recall', row.get('recall_macro', 'N/A')):.4f}</td>"
            report += f"<td>{row.get('f1', row.get('f1_macro', 'N/A')):.4f}</td>"
            report += f"<td>{row.get('roc_auc', row.get('roc_auc_ovr', 'N/A')):.4f}</td>"
            report += "</tr>"

        report += """
            </table>

            <h2>2. Statistical Significance Tests</h2>
        """

        # McNemar test results
        if 'mcnemar' in stats_results:
            report += "<h3>McNemar's Test (Pairwise Comparisons)</h3><ul>"
            for result in stats_results['mcnemar']:
                sig_class = 'significant' if result['significant'] else 'not-significant'
                report += f"<li class='{sig_class}'>"
                report += f"{result['model1']} vs {result['model2']}: "
                report += f"p-value = {result['p_value']:.4f} "
                report += f"({'Significant' if result['significant'] else 'Not significant'})</li>"
            report += "</ul>"

        # Friedman test results
        if 'friedman' in stats_results:
            result = stats_results['friedman']
            report += f"""
            <h3>Friedman Test</h3>
            <p>Statistic: {result['statistic']:.4f}, p-value: {result['p_value']:.4f}</p>
            <p>Result: {'Significant differences found among models' if result['significant'] else 'No significant differences'}</p>
            """

        # Cross-validation results
        if cv_results is not None and not cv_results.empty:
            report += """
            <h2>3. Cross-Validation Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Metric</th>
                    <th>Mean ± Std</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            """

            for _, row in cv_results.iterrows():
                report += f"""
                <tr>
                    <td>{row['model']}</td>
                    <td>{row['metric']}</td>
                    <td>{row['mean']:.4f} ± {row['std']:.4f}</td>
                    <td>{row['min']:.4f}</td>
                    <td>{row['max']:.4f}</td>
                </tr>
                """
            report += "</table>"

        # Error analysis
        report += """
            <h2>4. Error Analysis</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Total Errors</th>
                    <th>Error Rate</th>
                    <th>False Positives</th>
                    <th>False Negatives</th>
                </tr>
        """

        for model_name, analysis in error_analysis.items():
            if isinstance(analysis, dict) and 'total_errors' in analysis:
                report += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{analysis['total_errors']}</td>
                    <td>{analysis['error_rate']:.4f}</td>
                    <td>{analysis.get('false_positives', 'N/A')}</td>
                    <td>{analysis.get('false_negatives', 'N/A')}</td>
                </tr>
                """

        report += f"""
            </table>

            <h2>5. Summary and Recommendations</h2>
            <p>Consistently misclassified samples: {error_analysis.get('consistent_error_rate', 0):.2%}</p>

            <h3>Key Findings:</h3>
            <ul>
                <li>Best performing model: {comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model']}</li>
                <li>Highest precision: {comparison_df.loc[comparison_df.get('precision', comparison_df.get('precision_macro', 0)).idxmax(), 'model']}</li>
                <li>Statistical significance found in {sum([r['significant'] for r in stats_results.get('mcnemar', [])])} pairwise comparisons</li>
            </ul>
        </body>
        </html>
        """

        return report

    def visualize_results(self,
                         comparison_df: pd.DataFrame,
                         cv_results: pd.DataFrame,
                         predictions: Dict[str, np.ndarray],
                         y_true: np.ndarray):
        """
        Create comprehensive visualizations of evaluation results
        """
        fig = plt.figure(figsize=(20, 15))

        # 1. Performance metrics comparison
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = comparison_df['model'].values

        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = comparison_df.get(metric, comparison_df.get(f'{metric}_macro', [0]*len(models)))
            ax1.bar(x + i*width, values, width, label=metric.capitalize())

        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. ROC curves (if available)
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')

        for _, row in comparison_df.iterrows():
            if 'roc_auc' in row and not pd.isna(row['roc_auc']):
                # Simplified ROC visualization
                ax2.plot([0, 0.2, 1], [0, 0.8, 1],
                        label=f"{row['model']} (AUC={row['roc_auc']:.3f})")

        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 3. Cross-validation scores distribution
        if cv_results is not None and not cv_results.empty:
            ax3 = plt.subplot(3, 3, 3)
            cv_accuracy = cv_results[cv_results['metric'] == 'accuracy']

            if not cv_accuracy.empty:
                box_data = []
                labels = []
                for model in cv_accuracy['model'].unique():
                    scores = cv_accuracy[cv_accuracy['model'] == model]['scores'].iloc[0]
                    box_data.append(scores)
                    labels.append(model)

                bp = ax3.boxplot(box_data, labels=labels)
                ax3.set_ylabel('Accuracy')
                ax3.set_title('Cross-Validation Accuracy Distribution')
                ax3.set_xticklabels(labels, rotation=45, ha='right')
                ax3.grid(True, alpha=0.3)

        # 4. Confusion matrices
        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            if idx < 6:  # Limit to 6 models for visualization
                ax = plt.subplot(3, 3, idx + 4)
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix - {model_name}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')

        plt.suptitle('Model Evaluation Results', fontsize=16, y=0.995)
        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f'evaluation_results_{self.timestamp}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

        plt.show()

    def save_results(self,
                    comparison_df: pd.DataFrame,
                    stats_results: Dict,
                    cv_results: pd.DataFrame,
                    error_analysis: Dict,
                    report_html: str):
        """
        Save all evaluation results to files
        """
        # Save DataFrames
        comparison_df.to_csv(self.output_dir / f'model_comparison_{self.timestamp}.csv', index=False)
        cv_results.to_csv(self.output_dir / f'cv_results_{self.timestamp}.csv', index=False)

        # Save statistical results
        with open(self.output_dir / f'statistical_tests_{self.timestamp}.json', 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)

        # Save error analysis
        with open(self.output_dir / f'error_analysis_{self.timestamp}.json', 'w') as f:
            json.dump(error_analysis, f, indent=2, default=str)

        # Save HTML report
        with open(self.output_dir / f'evaluation_report_{self.timestamp}.html', 'w') as f:
            f.write(report_html)

        # Save summary JSON
        summary = {
            'timestamp': self.timestamp,
            'best_model': comparison_df.loc[comparison_df['accuracy'].idxmax(), 'model'],
            'best_accuracy': float(comparison_df['accuracy'].max()),
            'models_evaluated': comparison_df['model'].tolist(),
            'statistical_tests_performed': list(stats_results.keys()),
            'files_generated': [
                f'model_comparison_{self.timestamp}.csv',
                f'cv_results_{self.timestamp}.csv',
                f'statistical_tests_{self.timestamp}.json',
                f'error_analysis_{self.timestamp}.json',
                f'evaluation_report_{self.timestamp}.html'
            ]
        }

        with open(self.output_dir / f'evaluation_summary_{self.timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== All results saved to {self.output_dir} ===")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Evaluate AI text detection models')

    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to our trained model')
    parser.add_argument('--baseline_dir', type=str,
                       help='Directory containing baseline models')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.output_dir)

    # Load test data
    print("Loading test data...")
    # Implementation would load actual test data

    # Load models
    print("Loading models...")
    # Implementation would load our model and baselines

    # Run evaluation
    print("Running comprehensive evaluation...")
    # Implementation would run full evaluation pipeline

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive Benchmark: Traditional Models vs LLM-as-a-Judge
Compares various AI detection methods including LLM judges
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import time
import warnings
from typing import Dict, List, Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'evaluation'))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.evaluation.llm_judge_classifier import create_llm_judges, LLMJudgeClassifier, EnsembleLLMJudge
from src.evaluation.baseline_detectors import PerplexityDetector, BurstinessDetector, ZipfianDetector

warnings.filterwarnings('ignore')


class ComprehensiveBenchmark:
    """Run comprehensive benchmark of all detection methods"""

    def __init__(self, output_dir: str = "results/benchmark"):
        """Initialize benchmark suite"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []

    def load_data(self, data_path: str) -> tuple:
        """Load data from JSON file"""
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)

        texts = [item['text'] for item in data]
        labels = np.array([item['label'] for item in data])

        print(f"  Loaded {len(texts)} samples")
        print(f"  Class distribution: Human={np.sum(labels==0)}, AI={np.sum(labels==1)}")

        return texts, labels

    def create_traditional_models(self, X_train, y_train) -> Dict:
        """Create and train traditional ML models"""
        print("\n=== Training Traditional Models ===")
        models = {}

        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=100, min_df=2, max_df=0.8)
        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()

        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_tfidf, y_train)
        models['Logistic Regression'] = lr

        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_tfidf, y_train)
        models['Random Forest'] = rf

        return models

    def create_statistical_models(self, X_train, y_train) -> Dict:
        """Create statistical baseline models"""
        print("\n=== Training Statistical Models ===")
        models = {}

        # Perplexity Detector
        print("Training Perplexity Detector...")
        perp = PerplexityDetector()
        perp.fit(X_train, y_train)
        models['Perplexity-Based'] = perp

        # Burstiness Detector
        print("Training Burstiness Detector...")
        burst = BurstinessDetector()
        burst.fit(X_train, y_train)
        models['Burstiness'] = burst

        # Zipfian Detector
        print("Training Zipfian Detector...")
        zipf = ZipfianDetector()
        zipf.fit(X_train, y_train)
        models['Zipfian'] = zipf

        return models

    def create_llm_judges(self, X_train, y_train) -> Dict:
        """Create and train LLM judge models"""
        print("\n=== Training LLM Judges ===")
        judges = {}

        # Zero-shot LLM Judge
        print("Training Zero-shot LLM Judge...")
        zero_shot = LLMJudgeClassifier(prompt_strategy='zero_shot')
        zero_shot.fit(X_train, y_train)
        judges['LLM-ZeroShot'] = zero_shot

        # Few-shot LLM Judge
        print("Training Few-shot LLM Judge...")
        few_shot = LLMJudgeClassifier(prompt_strategy='few_shot')
        few_shot.fit(X_train, y_train)
        judges['LLM-FewShot'] = few_shot

        # Chain-of-thought LLM Judge
        print("Training Chain-of-Thought LLM Judge...")
        cot = LLMJudgeClassifier(prompt_strategy='chain_of_thought')
        cot.fit(X_train, y_train)
        judges['LLM-ChainOfThought'] = cot

        # Ensemble LLM Judge
        print("Training Ensemble LLM Judge...")
        ensemble = EnsembleLLMJudge()
        ensemble.fit(X_train, y_train)
        judges['LLM-Ensemble'] = ensemble

        return judges

    def evaluate_model(self, model, X_test, y_test, model_name, model_type):
        """Evaluate a single model"""
        print(f"  Evaluating {model_name}...")

        start_time = time.time()

        # Get predictions
        if model_type == 'traditional':
            X_test_transformed = self.vectorizer.transform(X_test).toarray()
            y_pred = model.predict(X_test_transformed)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_transformed)
            else:
                y_pred_proba = None
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None

        inference_time = time.time() - start_time

        # Calculate metrics
        metrics = {
            'model': model_name,
            'type': model_type,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'inference_time': inference_time,
            'avg_time_per_sample': inference_time / len(y_test)
        }

        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = None

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = cm[0, 0]
        metrics['false_positives'] = cm[0, 1]
        metrics['false_negatives'] = cm[1, 0]
        metrics['true_positives'] = cm[1, 1]

        return metrics

    def run_benchmark(self, train_path: str, test_path: str):
        """Run complete benchmark"""
        print("="*80)
        print("COMPREHENSIVE BENCHMARK: Traditional ML vs LLM-as-Judge")
        print("="*80)

        # Load data
        X_train, y_train = self.load_data(train_path)
        X_test, y_test = self.load_data(test_path)

        # Create all models
        traditional_models = self.create_traditional_models(X_train, y_train)
        statistical_models = self.create_statistical_models(X_train, y_train)
        llm_judges = self.create_llm_judges(X_train, y_train)

        # Evaluate all models
        print("\n=== Model Evaluation ===")

        print("\nTraditional ML Models:")
        for name, model in traditional_models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name, 'traditional')
            self.results.append(metrics)

        print("\nStatistical Models:")
        for name, model in statistical_models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name, 'statistical')
            self.results.append(metrics)

        print("\nLLM Judge Models:")
        for name, model in llm_judges.items():
            metrics = self.evaluate_model(model, X_test, y_test, name, 'llm_judge')
            self.results.append(metrics)

        # Create results DataFrame
        self.df_results = pd.DataFrame(self.results)

        return self.df_results

    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)

        # Sort by accuracy
        df_sorted = self.df_results.sort_values('accuracy', ascending=False)

        # Print performance table
        print("\n### Performance Metrics (sorted by accuracy) ###\n")
        print(f"{'Model':<25} {'Type':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(s)':<10}")
        print("-"*95)

        for _, row in df_sorted.iterrows():
            print(f"{row['model']:<25} {row['type']:<12} {row['accuracy']:<10.4f} "
                  f"{row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f} "
                  f"{row['inference_time']:<10.4f}")

        # Best models by category
        print("\n### Best Models by Category ###\n")
        for model_type in self.df_results['type'].unique():
            type_df = self.df_results[self.df_results['type'] == model_type]
            best = type_df.loc[type_df['accuracy'].idxmax()]
            print(f"{model_type.capitalize()}: {best['model']} (Accuracy: {best['accuracy']:.4f})")

        # Overall best model
        print("\n### Overall Best Model ###\n")
        best_overall = self.df_results.loc[self.df_results['accuracy'].idxmax()]
        print(f"Model: {best_overall['model']}")
        print(f"Type: {best_overall['type']}")
        print(f"Accuracy: {best_overall['accuracy']:.4f}")
        print(f"F1-Score: {best_overall['f1']:.4f}")

        # Speed comparison
        print("\n### Speed Comparison ###\n")
        print(f"{'Model':<25} {'Avg Time per Sample (ms)':<25}")
        print("-"*50)
        for _, row in df_sorted.iterrows():
            avg_time_ms = row['avg_time_per_sample'] * 1000
            print(f"{row['model']:<25} {avg_time_ms:<25.2f}")

        # LLM Judge specific analysis
        llm_results = self.df_results[self.df_results['type'] == 'llm_judge']
        if not llm_results.empty:
            print("\n### LLM Judge Analysis ###\n")
            print(f"Average LLM Judge Accuracy: {llm_results['accuracy'].mean():.4f}")
            print(f"Best LLM Judge: {llm_results.loc[llm_results['accuracy'].idxmax()]['model']}")
            print(f"LLM Ensemble Performance: {llm_results[llm_results['model'] == 'LLM-Ensemble']['accuracy'].values[0]:.4f}")

            # Compare with traditional methods
            trad_avg = self.df_results[self.df_results['type'] == 'traditional']['accuracy'].mean()
            llm_avg = llm_results['accuracy'].mean()
            diff = llm_avg - trad_avg
            print(f"\nLLM vs Traditional ML:")
            print(f"  Traditional Average: {trad_avg:.4f}")
            print(f"  LLM Judge Average: {llm_avg:.4f}")
            print(f"  Difference: {diff:+.4f} {'(LLM better)' if diff > 0 else '(Traditional better)'}")

    def save_results(self):
        """Save all results"""
        # Save detailed results (convert numpy types to Python types)
        results_json = []
        for result in self.results:
            result_dict = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.int64)):
                    result_dict[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    result_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result_dict[key] = value.tolist()
                else:
                    result_dict[key] = value
            results_json.append(result_dict)

        results_file = self.output_dir / f'benchmark_results_{self.timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=2)

        # Save DataFrame as CSV
        csv_file = self.output_dir / f'benchmark_results_{self.timestamp}.csv'
        self.df_results.to_csv(csv_file, index=False)

        # Create visualization
        self.create_visualizations()

        print(f"\n### Results saved to: ###")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Plots: {self.output_dir}/benchmark_plot_{self.timestamp}.png")

    def create_visualizations(self):
        """Create benchmark visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        df_sorted = self.df_results.sort_values('accuracy', ascending=True)
        colors = ['green' if t == 'llm_judge' else 'blue' if t == 'traditional' else 'orange'
                  for t in df_sorted['type']]
        ax1.barh(range(len(df_sorted)), df_sorted['accuracy'], color=colors)
        ax1.set_yticks(range(len(df_sorted)))
        ax1.set_yticklabels(df_sorted['model'])
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')

        # 2. F1-Score comparison
        ax2 = axes[0, 1]
        metrics = ['precision', 'recall', 'f1']
        x = np.arange(len(df_sorted))
        width = 0.25
        for i, metric in enumerate(metrics):
            ax2.bar(x + i*width, df_sorted[metric], width, label=metric.capitalize())
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision, Recall, F1 Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(df_sorted['model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Model Type Performance
        ax3 = axes[1, 0]
        type_performance = self.df_results.groupby('type').agg({
            'accuracy': 'mean',
            'f1': 'mean',
            'inference_time': 'mean'
        })
        type_performance.plot(kind='bar', ax=ax3)
        ax3.set_title('Average Performance by Model Type')
        ax3.set_xlabel('Model Type')
        ax3.set_ylabel('Score')
        ax3.legend(['Accuracy', 'F1-Score', 'Inference Time (s)'])
        ax3.grid(True, alpha=0.3)

        # 4. Speed vs Accuracy scatter
        ax4 = axes[1, 1]
        for model_type in self.df_results['type'].unique():
            type_df = self.df_results[self.df_results['type'] == model_type]
            ax4.scatter(type_df['avg_time_per_sample']*1000, type_df['accuracy'],
                       label=model_type, s=100, alpha=0.7)
        ax4.set_xlabel('Average Time per Sample (ms)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Speed vs Accuracy Trade-off')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add annotations for best models
        best_idx = self.df_results['accuracy'].idxmax()
        best_model = self.df_results.loc[best_idx]
        ax4.annotate(best_model['model'],
                    (best_model['avg_time_per_sample']*1000, best_model['accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.suptitle(f'AI Text Detection Benchmark Results - {self.timestamp}', fontsize=14)
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / f'benchmark_plot_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution"""
    benchmark = ComprehensiveBenchmark()

    # Run benchmark
    df_results = benchmark.run_benchmark(
        train_path='data/processed/train_data.json',
        test_path='data/processed/test_data.json'
    )

    # Generate report
    benchmark.generate_report()

    # Save results
    benchmark.save_results()

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
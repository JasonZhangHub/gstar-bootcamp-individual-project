# **A Hybrid Stylometric and Semantic Fluctuation Approach for Detecting AI-Generated Media News**

**Author:** Jiasheng (Jason) Zhang
**Date:** October 6, 2025
**GitHub:** [AI-Generated-News-Detection](https://github.com/JasonZhangHub/gstar-bootcamp-individual-project)

## **Executive Summary**

This project developed a hybrid detection system combining stylometric and semantic features to identify AI-generated news. The approach achieved **75% accuracy** and **78.3% F1-score**, outperforming GPT-based baselines by **+22.5%** and **+14.1%**. AI text exhibits shallower sentence structures, overly uniform coherence, and predictable grammar patterns.

## **1. Project Approach and Execution**

### 1.1 Research Motivation

LLMs enable mass production of synthetic news, threatening information integrity. Existing detectors rely on content-based classification, vulnerable to adversarial attacks. This project develops robust detection through hybrid stylometric-semantic analysis.

**Research Question:** Can hybrid stylometry and semantic analysis outperform standard LLM-based classifiers?

### 1.2 Methodology

The project followed a five-stage pipeline:

#### **Stage 1: Dataset Construction**
- Collected human-written news from multiple authoritative sources (AG News, Common Crawl News, Newsroom)
- Generated AI counterparts using state-of-the-art models (GPT-2-xl) on matched topics
- Created balanced datasets with 1:1 human-to-AI ratios to prevent class imbalance
- Final dataset: Combined human news corpus with synthetic AI-generated articles

#### **Stage 2: Feature Engineering - Stylometric Analysis**
Implemented traditional writing style analysis features:
- **Lexical diversity metrics**: Type-token ratio, hapax legomena frequency
- **Sentence complexity patterns**: Parse tree depth, dependency distances, syntactic structures
- **N-gram frequency distributions**: Unigram through trigram statistical patterns
- **Punctuation and formatting**: Capitalization, special character usage, sentence length variance

#### **Stage 3: Feature Engineering - Semantic Fluctuation Analysis (Novel Contribution)**
Developed innovative semantic coherence measures:
- **Topic coherence drift**: Semantic similarity tracking between consecutive paragraphs using sentence embeddings
- **Entity consistency scoring**: Named entity reference patterns throughout articles
- **Temporal logic patterns**: Event sequencing and temporal reference consistency
- **Source attribution patterns**: Citation frequency and specificity analysis
- **Factual grounding metrics**: Density ratio of verifiable claims vs. vague statements

#### **Stage 4: Model Development**
Trained multiple detection approaches:
- **Traditional ML classifiers**: Logistic Regression, Random Forest with hybrid features
- **LLM Judge baselines**: GPT-4o-mini zero-shot (100 samples)

#### **Stage 5: Evaluation and Benchmarking**
Comprehensive testing protocol:
- Cross-validation on balanced test sets
- Comparison against baseline methods including GPTZero-style approaches
- Feature importance analysis to identify discriminative patterns

## **2. Key Results, Findings, and Outcomes**

### 2.1 Model Performance

**Primary Model: Logistic Regression with Hybrid Features**

| Metric | Our Ensemble | GPT-Zero-Shot Baseline | Improvement |
|--------|--------------|------------------------|-------------|
| Accuracy | **75.0%** | 52.5% | **+22.5%** |
| Precision | **69.2%** | 51.5% | **+17.7%** |
| Recall | **90.0%** | 85.0% | **+5.0%** |
| F1-Score | **78.3%** | 64.2% | **+14.1%** |

### 2.2 Critical Findings: How AI Text Differs from Human Writing

**Top 5 Discriminative Features:**

1. **Complexity_7 (Importance: 36.5)** - AI text has **shallower sentence structures** with predictable syntactic patterns vs. human variation
2. **Embedding_3 (Importance: 29.8)** - AI shows **overly uniform topic coherence** across paragraphs vs. natural semantic drift
3. **Embedding_45 (Importance: 27.0)** - AI maintains **excessively consistent context** vs. human variation in flow
4. **Complexity_4 (Importance: 26.8)** - AI uses **predictable parse tree patterns** vs. diverse human constructions
5. **Complexity_3 (Importance: 26.8)** - AI has **narrow dependency distances** vs. complex human word relationships

### 2.3 Practical Implications

**Strengths:** High recall (90%) catches most AI content; balanced precision (69.2%) minimizes false positives; interpretable features; robust hybrid design resistant to single-vector attacks.

**Limitations:** Requires evaluation on adversarial paraphrasing and newer LLMs; continuous retraining needed as models evolve.

---

## **Conclusion**

This project demonstrated that hybrid stylometric-semantic analysis significantly outperforms existing baselines (**75% accuracy, 78.3% F1** vs. **52.5%, 64.2%**). AI text is distinguishable through reduced structural complexity, overly uniform coherence, and predictable grammar—reflecting LLMs' statistical nature vs. human variability. Robust detection requires multi-faceted analysis beyond simple perplexity metrics, with hybrid approaches essential as LLMs evolve.

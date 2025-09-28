#!/usr/bin/env python3
"""
Semantic Fluctuation Analysis for AI-Generated Text Detection
Analyzes semantic coherence, entity consistency, and factual patterns in text
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import re
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag

    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)

    NLP_AVAILABLE = True
except ImportError as e:
    NLP_AVAILABLE = False
    print(f"Warning: Some libraries not available: {e}")
    print("Install with: pip install spacy sentence-transformers torch scikit-learn")
    print("Then: python -m spacy download en_core_web_sm")


class SemanticFluctuationAnalyzer:
    """Analyze semantic fluctuations and coherence patterns in text"""

    def __init__(self,
                 spacy_model: str = "en_core_web_sm",
                 sentence_model: str = "all-MiniLM-L6-v2",
                 device: str = None):
        """
        Initialize the analyzer

        Args:
            spacy_model: SpaCy model for NLP tasks
            sentence_model: Sentence transformer model for embeddings
            device: Device for sentence transformer (None for auto-detect)
        """
        self.feature_names = []

        if NLP_AVAILABLE:
            # Load SpaCy
            try:
                self.nlp = spacy.load(spacy_model)
            except:
                print(f"Installing SpaCy model {spacy_model}...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", spacy_model])
                self.nlp = spacy.load(spacy_model)

            # Load sentence transformer
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            self.sentence_model = SentenceTransformer(sentence_model, device=device)
        else:
            self.nlp = None
            self.sentence_model = None

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all semantic fluctuation features

        Args:
            text: Input text

        Returns:
            Dictionary of feature names and values
        """
        features = {}

        # Clean text
        text = text.strip()
        if not text:
            return features

        # 1. Topic Coherence Drift
        coherence_features = self._analyze_topic_coherence(text)
        features.update(coherence_features)

        # 2. Entity Consistency Score
        entity_features = self._analyze_entity_consistency(text)
        features.update(entity_features)

        # 3. Temporal Logic Patterns
        temporal_features = self._analyze_temporal_logic(text)
        features.update(temporal_features)

        # 4. Source Attribution Patterns
        source_features = self._analyze_source_attribution(text)
        features.update(source_features)

        # 5. Factual Grounding Metrics
        factual_features = self._analyze_factual_grounding(text)
        features.update(factual_features)

        return features

    def _analyze_topic_coherence(self, text: str) -> Dict[str, float]:
        """
        Analyze topic coherence drift between consecutive segments
        Measures semantic similarity between paragraphs and sentences
        """
        features = {}

        if not self.sentence_model:
            return features

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        # If still no paragraphs, split into sentences
        if len(paragraphs) < 2:
            sentences = sent_tokenize(text)
            # Group sentences into pseudo-paragraphs (3 sentences each)
            paragraphs = []
            for i in range(0, len(sentences), 3):
                paragraphs.append(' '.join(sentences[i:i+3]))

        if len(paragraphs) > 1:
            # Get embeddings for paragraphs
            embeddings = self.sentence_model.encode(paragraphs)

            # Calculate coherence between consecutive paragraphs
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                coherence_scores.append(sim)

            if coherence_scores:
                features['coherence_mean'] = np.mean(coherence_scores)
                features['coherence_std'] = np.std(coherence_scores)
                features['coherence_min'] = np.min(coherence_scores)
                features['coherence_max'] = np.max(coherence_scores)

                # Detect sudden topic shifts (low similarity)
                topic_shifts = sum(1 for s in coherence_scores if s < 0.3)
                features['topic_shift_count'] = topic_shifts
                features['topic_shift_ratio'] = topic_shifts / len(coherence_scores)

                # Detect unusually high coherence (potential repetition)
                high_coherence = sum(1 for s in coherence_scores if s > 0.8)
                features['high_coherence_ratio'] = high_coherence / len(coherence_scores)

                # Coherence drift (how much coherence varies)
                if len(coherence_scores) > 1:
                    drift = np.abs(np.diff(coherence_scores))
                    features['coherence_drift_mean'] = np.mean(drift)
                    features['coherence_drift_max'] = np.max(drift)

        # Analyze sentence-level coherence within paragraphs
        sentence_coherences = []
        for paragraph in paragraphs[:5]:  # Limit for performance
            sentences = sent_tokenize(paragraph)
            if len(sentences) > 1:
                sent_embeddings = self.sentence_model.encode(sentences)
                for i in range(len(sent_embeddings) - 1):
                    sim = cosine_similarity([sent_embeddings[i]], [sent_embeddings[i+1]])[0][0]
                    sentence_coherences.append(sim)

        if sentence_coherences:
            features['sentence_coherence_mean'] = np.mean(sentence_coherences)
            features['sentence_coherence_std'] = np.std(sentence_coherences)

        return features

    def _analyze_entity_consistency(self, text: str) -> Dict[str, float]:
        """
        Analyze how consistently named entities are referenced
        Detects entity inconsistencies and reference patterns
        """
        features = {}

        if not self.nlp:
            return features

        # Process text with SpaCy
        doc = self.nlp(text[:1000000])  # Limit for performance

        # Extract entities and their contexts
        entity_mentions = defaultdict(list)
        entity_types = defaultdict(set)

        for ent in doc.ents:
            # Normalize entity text
            normalized = ent.text.lower().strip()
            entity_mentions[normalized].append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
            entity_types[normalized].add(ent.label_)

        if entity_mentions:
            # Entity frequency statistics
            mention_counts = [len(mentions) for mentions in entity_mentions.values()]
            features['entity_count'] = len(entity_mentions)
            features['entity_mention_mean'] = np.mean(mention_counts)
            features['entity_mention_std'] = np.std(mention_counts)
            features['entity_mention_max'] = np.max(mention_counts)

            # Entity type consistency (same entity, different types = inconsistency)
            type_inconsistencies = sum(1 for types in entity_types.values() if len(types) > 1)
            features['entity_type_inconsistency'] = type_inconsistencies
            features['entity_type_inconsistency_ratio'] = (
                type_inconsistencies / len(entity_mentions) if entity_mentions else 0
            )

            # Analyze entity distribution (how evenly entities are mentioned)
            total_mentions = sum(mention_counts)
            if total_mentions > 0:
                entity_distribution = [c / total_mentions for c in mention_counts]
                entropy = -sum(p * np.log2(p) for p in entity_distribution if p > 0)
                features['entity_distribution_entropy'] = entropy

            # Check for entity introduction patterns
            # Entities mentioned only once might indicate poor consistency
            single_mention = sum(1 for c in mention_counts if c == 1)
            features['single_mention_entity_ratio'] = single_mention / len(entity_mentions)

            # Analyze coreference patterns (pronouns after entity mentions)
            pronouns = [token for token in doc if token.pos_ == 'PRON']
            features['entity_pronoun_ratio'] = (
                len(pronouns) / len(entity_mentions) if entity_mentions else 0
            )

        # Person vs Organization vs Location distribution
        entity_type_counts = Counter(ent.label_ for ent in doc.ents)
        for ent_type in ['PERSON', 'ORG', 'GPE', 'LOC']:
            features[f'entity_{ent_type.lower()}_ratio'] = (
                entity_type_counts.get(ent_type, 0) / len(doc.ents) if doc.ents else 0
            )

        return features

    def _analyze_temporal_logic(self, text: str) -> Dict[str, float]:
        """
        Detect temporal inconsistencies and event sequencing patterns
        """
        features = {}

        # Temporal expression patterns
        temporal_patterns = {
            'past': r'\b(yesterday|ago|last\s+\w+|previous|earlier|before|was|were|had)\b',
            'present': r'\b(today|now|currently|presently|is|are|has|have)\b',
            'future': r'\b(tomorrow|next\s+\w+|will|shall|going\s+to|upcoming|later)\b',
            'specific_date': r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}|January|February|March|April|May|June|July|August|September|October|November|December)\b',
            'specific_time': r'\b(\d{1,2}:\d{2}|morning|afternoon|evening|night|noon|midnight)\b',
            'duration': r'\b(\d+\s*(hours?|minutes?|days?|weeks?|months?|years?))\b',
            'sequence': r'\b(first|second|third|then|next|finally|subsequently|meanwhile|during|after|before)\b'
        }

        temporal_counts = {}
        for temp_type, pattern in temporal_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            temporal_counts[temp_type] = len(matches)
            features[f'temporal_{temp_type}_count'] = len(matches)

        # Normalize by text length
        text_length = len(text.split())
        if text_length > 0:
            for temp_type in temporal_patterns:
                features[f'temporal_{temp_type}_density'] = temporal_counts[temp_type] / text_length

        # Temporal tense consistency
        sentences = sent_tokenize(text)
        if sentences and self.nlp:
            tense_shifts = 0
            prev_tense = None

            for sent in sentences[:50]:  # Limit for performance
                doc = self.nlp(sent)
                # Identify main verb tense
                verbs = [token for token in doc if token.pos_ == 'VERB']
                if verbs:
                    # Simple tense detection based on verb tags
                    main_verb = verbs[0]
                    if main_verb.tag_ in ['VBD', 'VBN']:
                        current_tense = 'past'
                    elif main_verb.tag_ in ['VBZ', 'VBP', 'VBG']:
                        current_tense = 'present'
                    elif main_verb.tag_ == 'VB' and any(
                        token.text in ['will', 'shall', 'going']
                        for token in doc if token.i < main_verb.i
                    ):
                        current_tense = 'future'
                    else:
                        current_tense = 'present'

                    if prev_tense and prev_tense != current_tense:
                        tense_shifts += 1
                    prev_tense = current_tense

            features['tense_shift_count'] = tense_shifts
            features['tense_shift_ratio'] = tense_shifts / len(sentences) if sentences else 0

        # Temporal ordering indicators
        total_temporal = sum(temporal_counts.values())
        if total_temporal > 0:
            features['temporal_specificity_ratio'] = (
                (temporal_counts['specific_date'] + temporal_counts['specific_time']) / total_temporal
            )
            features['temporal_sequence_ratio'] = temporal_counts['sequence'] / total_temporal

        return features

    def _analyze_source_attribution(self, text: str) -> Dict[str, float]:
        """
        Analyze patterns of source citations and quotes
        """
        features = {}

        # Quote patterns
        quote_patterns = {
            'direct_quote': r'"[^"]{10,}"',
            'said_verb': r'\b(said|says|stated|announced|reported|claimed|argued|noted|explained|told)\b',
            'according_to': r'\baccording\s+to\b',
            'source_indicator': r'\b(source|official|spokesperson|representative|expert|analyst|researcher)\b',
            'attribution_verb': r'\b(confirmed|revealed|disclosed|admitted|denied|suggested|indicated)\b',
            'anonymous_source': r'\b(anonymous|unnamed|undisclosed|sources?\s+close|sources?\s+familiar)\b',
            'specific_org': r'\b([A-Z][A-Za-z]+\s+(?:Corporation|Inc|LLC|Ltd|Company|Institute|University|Department|Agency|Bureau))\b'
        }

        quote_counts = {}
        for quote_type, pattern in quote_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE if quote_type != 'specific_org' else 0)
            quote_counts[quote_type] = len(matches)
            features[f'source_{quote_type}_count'] = len(matches)

        # Calculate quote density
        sentences = sent_tokenize(text)
        if sentences:
            features['quote_density'] = quote_counts['direct_quote'] / len(sentences)
            features['attribution_density'] = quote_counts['said_verb'] / len(sentences)

            # Sentences with attribution
            attributed_sentences = 0
            for sent in sentences:
                if any(re.search(pattern, sent, re.IGNORECASE)
                      for pattern in [quote_patterns['said_verb'],
                                    quote_patterns['according_to']]):
                    attributed_sentences += 1

            features['attributed_sentence_ratio'] = attributed_sentences / len(sentences)

        # Source specificity score
        total_attributions = quote_counts['said_verb'] + quote_counts['according_to']
        if total_attributions > 0:
            specific_attributions = quote_counts['specific_org']
            anonymous_attributions = quote_counts['anonymous_source']

            features['source_specificity_ratio'] = specific_attributions / total_attributions
            features['anonymous_source_ratio'] = anonymous_attributions / total_attributions

        # Quote length analysis
        quotes = re.findall(r'"([^"]{10,})"', text)
        if quotes:
            quote_lengths = [len(q.split()) for q in quotes]
            features['quote_length_mean'] = np.mean(quote_lengths)
            features['quote_length_std'] = np.std(quote_lengths)
            features['quote_length_max'] = np.max(quote_lengths)

        return features

    def _analyze_factual_grounding(self, text: str) -> Dict[str, float]:
        """
        Measure density and specificity of verifiable claims
        """
        features = {}

        # Factual indicator patterns
        factual_patterns = {
            'numbers': r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?\b',
            'statistics': r'\b(?:average|median|mean|percentage|rate|ratio|increase|decrease|growth|decline)\s+(?:of\s+)?\d+',
            'specific_claims': r'\b(?:study|research|survey|report|investigation|analysis)\s+(?:shows?|finds?|reveals?|indicates?|suggests?)',
            'hedging': r'\b(?:might|could|possibly|perhaps|maybe|seems?|appears?|likely|unlikely|probably|generally|usually|often|sometimes)\b',
            'certainty': r'\b(?:definitely|certainly|clearly|obviously|undoubtedly|surely|absolutely|always|never|must|will)\b',
            'vague_quantifiers': r'\b(?:many|few|some|several|various|numerous|multiple|lots?\s+of|a\s+number\s+of)\b',
            'specific_quantifiers': r'\b(?:all|none|every|no|each|any|both|either|neither|\d+\s+(?:percent|%))\b'
        }

        factual_counts = {}
        text_words = len(text.split())

        for fact_type, pattern in factual_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            factual_counts[fact_type] = len(matches)
            features[f'factual_{fact_type}_count'] = len(matches)

            # Density (normalized by text length)
            if text_words > 0:
                features[f'factual_{fact_type}_density'] = len(matches) / text_words

        # Factual specificity score
        if text_words > 0:
            specific_indicators = (
                factual_counts['numbers'] +
                factual_counts['statistics'] +
                factual_counts['specific_claims'] +
                factual_counts['specific_quantifiers']
            )
            vague_indicators = (
                factual_counts['hedging'] +
                factual_counts['vague_quantifiers']
            )

            features['factual_specificity_score'] = specific_indicators / text_words
            features['factual_vagueness_score'] = vague_indicators / text_words

            if (specific_indicators + vague_indicators) > 0:
                features['factual_specificity_ratio'] = (
                    specific_indicators / (specific_indicators + vague_indicators)
                )

        # Certainty vs hedging balance
        total_modality = factual_counts['certainty'] + factual_counts['hedging']
        if total_modality > 0:
            features['certainty_ratio'] = factual_counts['certainty'] / total_modality
            features['hedging_ratio'] = factual_counts['hedging'] / total_modality

        # Analyze claim-evidence patterns
        sentences = sent_tokenize(text)
        if sentences:
            claim_sentences = 0
            evidence_sentences = 0

            for sent in sentences:
                # Check for claim indicators
                if re.search(r'\b(?:shows?|proves?|demonstrates?|confirms?|reveals?)\b', sent, re.IGNORECASE):
                    claim_sentences += 1

                # Check for evidence indicators
                if re.search(r'\b(?:according|based\s+on|evidence|data|study|research|survey|report)\b', sent, re.IGNORECASE):
                    evidence_sentences += 1

            features['claim_sentence_ratio'] = claim_sentences / len(sentences)
            features['evidence_sentence_ratio'] = evidence_sentences / len(sentences)

            # Claim-evidence balance
            if claim_sentences > 0:
                features['evidence_per_claim'] = evidence_sentences / claim_sentences

        return features


def process_dataset(input_file: str, output_dir: str, sample_size: int = None):
    """
    Process a dataset and extract semantic fluctuation features

    Args:
        input_file: Path to input JSON file
        output_dir: Output directory for features
        sample_size: Optional limit for testing
    """
    # Initialize analyzer
    print("Initializing Semantic Fluctuation Analyzer...")
    analyzer = SemanticFluctuationAnalyzer()

    # Load data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        samples = data
    elif 'train' in data and 'test' in data:
        samples = data['train'] + data['test']
    elif 'articles' in data:
        samples = data['articles']
    else:
        print("Unexpected JSON structure")
        return

    if sample_size:
        samples = samples[:sample_size]

    print(f"Processing {len(samples)} samples...")

    # Extract features
    features_list = []
    labels = []

    for sample in tqdm(samples, desc="Extracting semantic features"):
        # Get text
        if isinstance(sample, dict):
            text = sample.get('text', sample.get('article', sample.get('document', '')))
            label = sample.get('label', sample.get('is_ai', -1))
        else:
            text = str(sample)
            label = -1

        if not text:
            continue

        # Extract features
        try:
            features = analyzer.extract_features(text)
            features_list.append(features)
            labels.append(label)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    if not features_list:
        print("No features extracted")
        return None

    # Create DataFrame
    df_features = pd.DataFrame(features_list)
    df_features['label'] = labels

    # Fill missing values with 0
    df_features = df_features.fillna(0)

    # Save features
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_file = output_path / f"semantic_features_{Path(input_file).stem}.csv"
    df_features.to_csv(csv_file, index=False)
    print(f"Saved features to {csv_file}")

    # Save as numpy arrays for ML
    X = df_features.drop('label', axis=1).values
    y = df_features['label'].values

    np_file = output_path / f"semantic_features_{Path(input_file).stem}.npz"
    np.savez(np_file, X=X, y=y, feature_names=df_features.columns[:-1].tolist())
    print(f"Saved numpy arrays to {np_file}")

    # Print feature statistics
    print(f"\nFeature Statistics:")
    print(f"  Number of samples: {len(df_features)}")
    print(f"  Number of features: {len(df_features.columns) - 1}")
    print(f"  Label distribution: {Counter(labels)}")

    # Print top features by variance
    feature_vars = df_features.drop('label', axis=1).var().sort_values(ascending=False)
    print(f"\nTop 10 features by variance:")
    for feat, var in feature_vars.head(10).items():
        print(f"  {feat}: {var:.4f}")

    # Print feature category summaries
    print(f"\nFeature Categories:")
    categories = {
        'Coherence': 'coherence_',
        'Entity': 'entity_',
        'Temporal': 'temporal_',
        'Source': 'source_',
        'Factual': 'factual_'
    }

    for cat_name, prefix in categories.items():
        cat_features = [col for col in df_features.columns if col.startswith(prefix)]
        print(f"  {cat_name}: {len(cat_features)} features")

    return df_features


def main():
    parser = argparse.ArgumentParser(description='Extract semantic fluctuation features')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with text data')
    parser.add_argument('--output', type=str, default='../../features',
                       help='Output directory for features')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing')

    args = parser.parse_args()

    # Process dataset
    process_dataset(args.input, args.output, args.sample)


if __name__ == "__main__":
    # If no arguments, run test
    import sys
    if len(sys.argv) == 1:
        print("Running test with sample data...")

        # Create test data
        test_data = [
            {
                "text": """
                According to John Smith from ABC Corporation, the quarterly revenue increased by 25% last year.
                "This represents significant growth," Smith said yesterday. The company reported earnings of $2.3 billion.

                However, tomorrow the situation might change. ABC Corporation was founded in 1950. Smith, who is now
                the CEO, stated that next week will be crucial. The revenue in 2019 was only $1 billion.

                Some analysts believe the growth could continue. Many experts suggest various factors contributed.
                A recent study shows definitive proof of market expansion. The research indicates strong performance.
                """,
                "label": 0
            },
            {
                "text": """
                The technology is advancing rapidly. Many people are using it. It has various applications.

                Scientists made a discovery. The discovery is important. It could change things. Some researchers
                think it might work. Others are not sure. Perhaps it will be successful.

                The company announced results. Results were positive. Growth happened. Markets responded well.
                Analysts said something. The future looks interesting. More developments are coming.
                """,
                "label": 1
            }
        ]

        # Save test data
        with open('/tmp/test_semantic.json', 'w') as f:
            json.dump(test_data, f)

        # Process
        df = process_dataset('/tmp/test_semantic.json', '/tmp/')
        print("\nTest completed successfully!")
    else:
        main()
#!/usr/bin/env python3
"""
Generate equal number of AI articles for each human article in the dataset
Creates a balanced dataset with 1:1 ratio of human to AI-generated articles
"""

import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import argparse
import random

class EqualAIGenerator:
    """Generate AI counterparts for each human article"""

    def __init__(self, model_name: str = "gpt2", batch_size: int = 10):
        """
        Initialize generator

        Args:
            model_name: Model to use (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
            batch_size: Number of articles to save at once
        """
        print(f"Initializing {model_name} model...")
        self.model_name = model_name
        self.batch_size = batch_size
        self.generator = pipeline('text-generation', model=model_name)
        print("✓ Model loaded")

    def create_prompt(self, article_text: str, category: str) -> str:
        """
        Create a prompt for AI generation based on the article

        Args:
            article_text: Original article text
            category: Article category

        Returns:
            Prompt string
        """
        # Extract key information from the article
        sentences = article_text.split('.')
        first_sentence = sentences[0].strip() + "."

        # Create category-specific prompts
        category_prompts = {
            'World': "Breaking News: ",
            'Sports': "Sports Update: ",
            'Business': "Business News: ",
            'Technology': "Tech News: "
        }

        # Build prompt with context
        prompt_prefix = category_prompts.get(category, "News: ")

        # Use first sentence as seed, add context
        if len(sentences) > 1:
            # Extract topic from first sentence
            prompt = f"{prompt_prefix}{first_sentence} According to reports,"
        else:
            prompt = f"{prompt_prefix}{first_sentence}"

        return prompt

    def generate_ai_article(self, original: dict) -> dict:
        """
        Generate AI counterpart for a single article

        Args:
            original: Original article dictionary

        Returns:
            AI-generated article dictionary
        """
        text = original.get('text', '')
        category = original.get('label_name', 'news')

        # Create prompt
        prompt = self.create_prompt(text, category)

        # Calculate target length
        word_count = len(text.split())
        max_length = min(word_count * 2, 512)  # GPT-2 token limit

        # Generate
        result = self.generator(
            prompt,
            max_length=max_length,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=50256
        )

        ai_text = result[0]['generated_text']

        # Remove prompt from generated text
        if prompt in ai_text:
            ai_text = ai_text.replace(prompt, "", 1).strip()

        # Create AI article object
        ai_article = {
            'text': ai_text,
            'label': original.get('label'),
            'label_name': category,
            'source': 'ai_generated',
            'model': self.model_name,
            'is_ai': True,
            'original_id': original.get('id', hashlib.md5(text.encode()).hexdigest()[:8]),
            'generation_date': datetime.now().isoformat(),
            'length': len(ai_text)
        }

        return ai_article

    def process_dataset(self, input_file: str, output_dir: str, resume_from: int = 0):
        """
        Process entire dataset to generate AI counterparts

        Args:
            input_file: Path to AG News JSON file
            output_dir: Output directory for generated articles
            resume_from: Resume from this index (for interrupted processing)
        """
        # Load dataset
        print(f"Loading dataset from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each split
        all_ai_articles = []
        all_human_articles = []

        for split in ['train', 'test']:
            if split not in data:
                continue

            articles = data[split]
            print(f"\nProcessing {split} set: {len(articles)} articles")

            ai_articles = []
            human_articles = []
            errors = 0

            # Resume from specified index
            start_idx = resume_from if split == 'train' else 0

            # Process with progress bar
            for i, article in enumerate(tqdm(articles[start_idx:],
                                           desc=f"Generating {split}",
                                           initial=start_idx)):
                try:
                    # Keep human article
                    human_article = article.copy()
                    human_article['is_ai'] = False
                    human_article['source'] = 'ag_news'
                    human_articles.append(human_article)

                    # Generate AI counterpart
                    ai_article = self.generate_ai_article(article)
                    ai_articles.append(ai_article)

                    # Save batch periodically
                    if (i + 1) % self.batch_size == 0:
                        self._save_batch(ai_articles, human_articles, output_path,
                                       split, i + start_idx + 1)

                except Exception as e:
                    errors += 1
                    if errors % 10 == 0:
                        print(f"\nErrors so far: {errors}")
                    continue

                # Rate limiting to avoid overload
                if self.model_name in ['gpt2-xl', 'gpt2-large']:
                    time.sleep(0.1)

            # Save final batch
            if ai_articles:
                self._save_batch(ai_articles, human_articles, output_path,
                               split, len(articles))

            all_ai_articles.extend(ai_articles)
            all_human_articles.extend(human_articles)

            print(f"✓ {split}: Generated {len(ai_articles)} AI articles, {errors} errors")

        # Create combined balanced dataset
        self._create_balanced_dataset(all_human_articles, all_ai_articles, output_path)

        print(f"\n✓ Processing complete!")
        print(f"  Total human articles: {len(all_human_articles)}")
        print(f"  Total AI articles: {len(all_ai_articles)}")

    def _save_batch(self, ai_articles: list, human_articles: list,
                   output_path: Path, split: str, count: int):
        """Save batch of articles"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save AI articles
        ai_file = output_path / f"ai_{split}_{self.model_name}_{count}_articles.json"
        with open(ai_file, 'w', encoding='utf-8') as f:
            json.dump(ai_articles, f, indent=2, ensure_ascii=False)

        # Save human articles
        human_file = output_path / f"human_{split}_{count}_articles.json"
        with open(human_file, 'w', encoding='utf-8') as f:
            json.dump(human_articles, f, indent=2, ensure_ascii=False)

    def _create_balanced_dataset(self, human_articles: list, ai_articles: list,
                                output_path: Path):
        """Create balanced dataset for training"""
        print("\nCreating balanced dataset...")

        # Combine and shuffle
        combined = []

        for human in human_articles:
            combined.append({
                'text': human['text'],
                'label': 0,  # 0 for human
                'label_name': 'human',
                'category': human.get('label_name', 'unknown'),
                'source': 'ag_news'
            })

        for ai in ai_articles:
            combined.append({
                'text': ai['text'],
                'label': 1,  # 1 for AI
                'label_name': 'ai_generated',
                'category': ai.get('label_name', 'unknown'),
                'source': ai['model']
            })

        # Shuffle
        random.shuffle(combined)

        # Split into train/test (80/20)
        split_idx = int(len(combined) * 0.8)
        train_data = combined[:split_idx]
        test_data = combined[split_idx:]

        # Save balanced dataset
        balanced_file = output_path / f"balanced_dataset_{self.model_name}.json"
        with open(balanced_file, 'w', encoding='utf-8') as f:
            json.dump({
                'train': train_data,
                'test': test_data,
                'metadata': {
                    'total_samples': len(combined),
                    'human_samples': len(human_articles),
                    'ai_samples': len(ai_articles),
                    'model': self.model_name,
                    'generation_date': datetime.now().isoformat()
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved balanced dataset: {balanced_file}")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Test: {len(test_data)} samples")

        # Save CSV version for easy inspection
        import pandas as pd
        df = pd.DataFrame(combined)
        csv_file = output_path / f"balanced_dataset_{self.model_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Saved CSV: {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate equal AI counterparts for AG News')
    parser.add_argument('--input', type=str,
                       default='../../data/datasets/ag_news_full.json',
                       help='Input AG News JSON file')
    parser.add_argument('--output', type=str,
                       default='../../data/ai_generated',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='gpt2',
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                       help='Model to use')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for saving')
    parser.add_argument('--resume-from', type=int, default=0,
                       help='Resume from this index (if interrupted)')

    args = parser.parse_args()

    # Generate AI counterparts
    generator = EqualAIGenerator(args.model, args.batch_size)
    generator.process_dataset(args.input, args.output, args.resume_from)

if __name__ == "__main__":
    main()
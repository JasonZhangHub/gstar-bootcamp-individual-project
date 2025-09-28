#!/usr/bin/env python3
"""
Simple GPT-2 article generation using transformers pipeline
Fast and easy approach for testing
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

def generate_ai_articles(input_file: str, output_dir: str, model_name: str = "gpt2-xl", limit: int = 10):
    """Generate AI articles using GPT-2 models"""

    print(f"Initializing {model_name} model...")
    generator = pipeline('text-generation', model=model_name)
    print("✓ Model loaded")

    # Load human articles
    print(f"\nLoading articles from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        articles = data[:limit]
    else:
        articles = data.get('articles', data.get('train', []))[:limit]

    print(f"Loaded {len(articles)} articles")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process articles
    ai_articles = []

    print("\nGenerating AI counterparts...")
    for i, article in enumerate(tqdm(articles, desc="Processing")):
        try:
            # Extract text
            if isinstance(article, dict):
                original_text = article.get('text', article.get('article', ''))
                source = article.get('source', 'unknown')
                category = article.get('category', article.get('label_name', 'news'))
            else:
                original_text = str(article)
                source = 'unknown'
                category = 'news'

            if not original_text or len(original_text) < 50:
                continue

            # Create prompt (use first sentence as seed)
            sentences = original_text.split('.')
            prompt = sentences[0] + "."

            # For news rewriting, add context
            if len(sentences) > 1:
                prompt = f"News: {prompt} According to reports,"

            # Generate
            word_count = len(original_text.split())
            max_length = min(word_count * 2, 512)  # GPT-2 limit

            result = generator(
                prompt,
                max_length=max_length,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=50256  # Set pad token to avoid warning
            )

            ai_text = result[0]['generated_text']

            # Remove the prompt from generated text
            if prompt in ai_text:
                ai_text = ai_text.replace(prompt, "", 1).strip()

            # Ensure minimum length
            if ai_text and len(ai_text) > 50:
                ai_articles.append({
                    'text': ai_text,
                    'original_text': original_text,
                    'source': source,
                    'category': category,
                    'model': model_name,
                    'is_ai_generated': True,
                    'generation_date': datetime.now().isoformat(),
                    'original_length': len(original_text),
                    'ai_length': len(ai_text),
                    'id': hashlib.md5(ai_text.encode()).hexdigest()[:8]
                })

        except Exception as e:
            print(f"\nError processing article {i}: {e}")
            continue

    # Save results
    output_file = output_path / f"ai_articles_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ai_articles, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {len(ai_articles)} AI articles")
    print(f"  Saved to: {output_file}")

    # Save summary
    summary = {
        'model': model_name,
        'total_generated': len(ai_articles),
        'generation_date': datetime.now().isoformat(),
        'avg_length': sum(a['ai_length'] for a in ai_articles) / len(ai_articles) if ai_articles else 0,
        'categories': {}
    }

    for article in ai_articles:
        cat = article.get('category', 'unknown')
        summary['categories'][cat] = summary['categories'].get(cat, 0) + 1

    summary_file = output_path / f"summary_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Print sample
    if ai_articles:
        print("\nSample AI-generated text (first 500 chars):")
        print("-" * 50)
        print(ai_articles[0]['text'][:500])
        print("-" * 50)

    return ai_articles

if __name__ == "__main__":
    # Test with standard GPT-2 (faster on CPU)
    generate_ai_articles(
        input_file="data/datasets/combined_human_news_full.json",
        output_dir="data/ai_generated",
        model_name="gpt2",  # Using base GPT-2 for speed. Can also use "gpt2-medium", "gpt2-large", "gpt2-xl"
        limit=10
    )
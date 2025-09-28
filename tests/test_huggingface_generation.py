#!/usr/bin/env python3
"""
Simple test script for AI article generation using HuggingFace Inference API
Free and fast alternative to local model loading
"""

import json
import time
from pathlib import Path
from huggingface_hub import InferenceClient
import hashlib
from datetime import datetime

def generate_ai_articles_hf(input_file: str, output_dir: str, limit: int = 10):
    """Generate AI articles using HuggingFace Inference API (free tier)"""

    # Initialize HF Inference Client (no token needed for free tier)
    # Use a model available on the free inference API
    client = InferenceClient("gpt2")

    # Load human articles
    print(f"Loading articles from {input_file}...")
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
    for i, article in enumerate(articles):
        try:
            # Extract text
            if isinstance(article, dict):
                original_text = article.get('text', article.get('article', ''))
                source = article.get('source', 'unknown')
                category = article.get('category', 'news')
            else:
                original_text = str(article)
                source = 'unknown'
                category = 'news'

            if not original_text or len(original_text) < 50:
                continue

            # Truncate very long articles for the prompt
            truncated = original_text[:1500] if len(original_text) > 1500 else original_text
            word_count = len(original_text.split())

            # Create prompt
            prompt = f"""Rewrite the following news article in your own words.
Keep the same general topic and information, but use different phrasing and structure.
Target length: approximately {word_count} words.

Original article:
{truncated}

Your rewritten version:"""

            print(f"  [{i+1}/{len(articles)}] Generating article (source: {source})...")

            # Generate using HF Inference API
            response = client.text_generation(
                prompt,
                max_new_tokens=min(word_count * 2, 1024),
                temperature=0.8,
                top_p=0.9,
                return_full_text=False
            )

            # Clean the response
            ai_text = response.strip()

            if ai_text and len(ai_text) > 50:
                ai_articles.append({
                    'text': ai_text,
                    'original_text': original_text,
                    'source': source,
                    'category': category,
                    'model': 'zephyr-7b-beta',
                    'is_ai_generated': True,
                    'generation_date': datetime.now().isoformat(),
                    'original_length': len(original_text),
                    'ai_length': len(ai_text),
                    'id': hashlib.md5(ai_text.encode()).hexdigest()[:8]
                })
                print(f"    ✓ Generated {len(ai_text)} chars (original: {len(original_text)} chars)")
            else:
                print(f"    ✗ Generation failed or too short")

            # Rate limiting for free tier
            time.sleep(2)

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    # Save results
    if ai_articles:
        output_file = output_path / f"ai_articles_hf_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ai_articles, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Successfully generated {len(ai_articles)} AI articles")
        print(f"  Saved to: {output_file}")

        # Print sample
        print("\nSample AI-generated text (first 500 chars):")
        print("-" * 50)
        print(ai_articles[0]['text'][:500])
        print("-" * 50)
    else:
        print("\n✗ No articles were generated successfully")

if __name__ == "__main__":
    # Test with 10 articles using HF Inference API
    generate_ai_articles_hf(
        input_file="data/datasets/combined_human_news_full.json",
        output_dir="data/ai_generated",
        limit=10
    )
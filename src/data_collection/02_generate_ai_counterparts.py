#!/usr/bin/env python3
"""
Generate AI counterparts for human-written news articles
Supports multiple models including open-source options for local testing
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import hashlib
from datetime import datetime

# Optional imports for different model providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Note: OpenAI not installed. Run: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Note: Anthropic not installed. Run: pip install anthropic")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: Transformers not installed. Run: pip install transformers torch")

try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False
    print("Note: Hugging Face Hub not installed. Run: pip install huggingface-hub")


class AIArticleGenerator:
    """Generate AI versions of human-written articles using various models"""

    def __init__(self, model_name: str = "mistral-7b", api_key: Optional[str] = None):
        """
        Initialize the generator with specified model

        Args:
            model_name: Model to use (gpt-4, claude, mistral-7b, llama2-7b, etc.)
            api_key: API key for commercial models
        """
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.model = None
        self.tokenizer = None

        # Model configurations
        self.model_configs = {
            # Open-source models (recommended for local testing - no auth required)
            'phi-2': {
                'model_id': 'microsoft/phi-2',
                'type': 'local',
                'description': 'Small but capable, low resource, no auth needed',
                'vram_required': '6GB',
                'quality': 'good'
            },
            'gpt2-large': {
                'model_id': 'gpt2-large',
                'type': 'local',
                'description': 'Classic GPT-2, no auth required',
                'vram_required': '4GB',
                'quality': 'good'
            },
            'gpt2-xl': {
                'model_id': 'gpt2-xl',
                'type': 'local',
                'description': 'Larger GPT-2, better quality',
                'vram_required': '8GB',
                'quality': 'very good'
            },
            'distilgpt2': {
                'model_id': 'distilgpt2',
                'type': 'local',
                'description': 'Lightweight, fast, low resource',
                'vram_required': '2GB',
                'quality': 'decent'
            },
            'bloom-560m': {
                'model_id': 'bigscience/bloom-560m',
                'type': 'local',
                'description': 'Multilingual, small, no auth',
                'vram_required': '2GB',
                'quality': 'decent'
            },
            'bloom-1b7': {
                'model_id': 'bigscience/bloom-1b7',
                'type': 'local',
                'description': 'Larger BLOOM, better quality',
                'vram_required': '4GB',
                'quality': 'good'
            },
            'opt-1.3b': {
                'model_id': 'facebook/opt-1.3b',
                'type': 'local',
                'description': 'Meta OPT model, good quality',
                'vram_required': '4GB',
                'quality': 'good'
            },
            'opt-2.7b': {
                'model_id': 'facebook/opt-2.7b',
                'type': 'local',
                'description': 'Larger OPT, better performance',
                'vram_required': '8GB',
                'quality': 'very good'
            },
            # Models requiring authentication (need HF token)
            'mistral-7b': {
                'model_id': 'mistralai/Mistral-7B-Instruct-v0.2',
                'type': 'local',
                'description': 'Excellent quality (requires HF auth)',
                'vram_required': '16GB',
                'quality': 'excellent',
                'auth_required': True
            },
            'llama2-7b': {
                'model_id': 'meta-llama/Llama-2-7b-chat-hf',
                'type': 'local',
                'description': 'Meta Llama 2 (requires HF auth)',
                'vram_required': '16GB',
                'quality': 'very good',
                'auth_required': True
            },
            'zephyr-7b': {
                'model_id': 'HuggingFaceH4/zephyr-7b-beta',
                'type': 'local',
                'description': 'Fine-tuned Mistral (may require auth)',
                'vram_required': '16GB',
                'quality': 'excellent',
                'auth_required': True
            },
            # Hugging Face Inference API (free tier available)
            'hf-mistral': {
                'model_id': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'type': 'hf-inference',
                'description': 'Via HF Inference API (free tier)',
                'quality': 'excellent'
            },
            'hf-llama': {
                'model_id': 'meta-llama/Llama-2-70b-chat-hf',
                'type': 'hf-inference',
                'description': 'Via HF Inference API',
                'quality': 'excellent'
            },
            # Commercial APIs
            'gpt-3.5': {
                'model_id': 'gpt-3.5-turbo',
                'type': 'openai',
                'description': 'OpenAI GPT-3.5',
                'quality': 'very good'
            },
            'gpt-4': {
                'model_id': 'gpt-4-turbo-preview',
                'type': 'openai',
                'description': 'OpenAI GPT-4',
                'quality': 'state-of-the-art'
            },
            'claude-3': {
                'model_id': 'claude-3-sonnet-20240229',
                'type': 'anthropic',
                'description': 'Anthropic Claude 3',
                'quality': 'state-of-the-art'
            }
        }

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_name not in self.model_configs:
            print(f"Warning: Unknown model {self.model_name}, defaulting to mistral-7b")
            self.model_name = 'mistral-7b'

        config = self.model_configs[self.model_name]
        print(f"\nInitializing {self.model_name}...")
        print(f"  Description: {config['description']}")
        print(f"  Quality: {config['quality']}")

        if config['type'] == 'local':
            self._init_local_model(config['model_id'])
        elif config['type'] == 'hf-inference':
            self._init_hf_inference(config['model_id'])
        elif config['type'] == 'openai':
            self._init_openai(config['model_id'])
        elif config['type'] == 'anthropic':
            self._init_anthropic(config['model_id'])

    def _init_local_model(self, model_id: str):
        """Initialize local transformers model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not installed. Run: pip install transformers torch")

        print(f"Loading local model: {model_id}")
        print("This may take a few minutes on first run...")

        try:
            # Use quantization to reduce memory usage
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Try to use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            if device == "cuda":
                # Load with 8-bit quantization to save memory
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except:
                    # Fallback to regular loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                # CPU loading (will be slower)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                except:
                    # Fallback for older transformers versions
                    self.model = AutoModelForCausalLM.from_pretrained(model_id)

            print(f"✓ Model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTip: For local testing without GPU, consider using:")
            print("  - phi-2 (6GB RAM)")
            print("  - HF Inference API (hf-mistral)")
            raise

    def _init_hf_inference(self, model_id: str):
        """Initialize Hugging Face Inference API"""
        if not HF_INFERENCE_AVAILABLE:
            raise ImportError("HF Hub not installed. Run: pip install huggingface-hub")

        # Use API key if provided, otherwise use free tier
        token = self.api_key or os.getenv("HUGGINGFACE_TOKEN")

        print(f"Using HF Inference API for {model_id}")
        if not token:
            print("Note: Using free tier (rate limited). Set HUGGINGFACE_TOKEN for higher limits.")

        self.model = InferenceClient(model_id, token=token)

    def _init_openai(self, model_id: str):
        """Initialize OpenAI API"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Run: pip install openai")

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = openai.OpenAI(api_key=api_key)
        self.model_id = model_id

    def _init_anthropic(self, model_id: str):
        """Initialize Anthropic API"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")

        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        self.model = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id

    def generate_article(self, original_article: str, preserve_length: bool = True) -> str:
        """
        Generate AI version of an article

        Args:
            original_article: The human-written article
            preserve_length: Try to match original article length

        Returns:
            AI-generated article
        """
        # Create prompt for rewriting
        word_count = len(original_article.split())

        prompt = f"""Rewrite the following news article in your own words.
Keep the same general topic and information, but use different phrasing and structure.
Target length: approximately {word_count} words.

Original article:
{original_article[:2000]}  # Truncate very long articles for prompt

Your rewritten version:"""

        config = self.model_configs[self.model_name]

        if config['type'] == 'local':
            return self._generate_local(prompt, word_count)
        elif config['type'] == 'hf-inference':
            return self._generate_hf(prompt, word_count)
        elif config['type'] == 'openai':
            return self._generate_openai(prompt, word_count)
        elif config['type'] == 'anthropic':
            return self._generate_anthropic(prompt, word_count)

    def _generate_local(self, prompt: str, target_words: int) -> str:
        """Generate using local model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        # Move to same device as model
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # Generate with appropriate length
        max_new_tokens = min(target_words * 2, 1024)  # Approximate tokens from words

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part
        if prompt in response:
            response = response.split(prompt)[-1].strip()

        return response

    def _generate_hf(self, prompt: str, target_words: int) -> str:
        """Generate using HF Inference API"""
        try:
            response = self.model.text_generation(
                prompt,
                max_new_tokens=target_words * 2,
                temperature=0.8,
                top_p=0.9,
                return_full_text=False
            )
            return response
        except Exception as e:
            print(f"HF Inference error: {e}")
            return ""

    def _generate_openai(self, prompt: str, target_words: int) -> str:
        """Generate using OpenAI API"""
        try:
            response = self.model.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=target_words * 2,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return ""

    def _generate_anthropic(self, prompt: str, target_words: int) -> str:
        """Generate using Anthropic API"""
        try:
            response = self.model.messages.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=target_words * 2,
                temperature=0.8
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic error: {e}")
            return ""


def process_dataset(
    input_file: str,
    output_dir: str,
    model_name: str = "mistral-7b",
    limit: Optional[int] = None,
    api_key: Optional[str] = None
):
    """
    Process a dataset of human articles to generate AI counterparts

    Args:
        input_file: Path to JSON file with human articles
        output_dir: Directory to save AI-generated articles
        model_name: Model to use for generation
        limit: Maximum number of articles to process
        api_key: API key for commercial models
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load human articles
    print(f"\nLoading articles from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, list):
        articles = data
    elif 'articles' in data:
        articles = data['articles']
    elif 'train' in data:
        articles = data['train']
    else:
        print("Unexpected JSON structure")
        return

    if limit:
        articles = articles[:limit]

    print(f"Loaded {len(articles)} articles")

    # Initialize generator
    generator = AIArticleGenerator(model_name, api_key)

    # Process articles
    ai_articles = []
    failed_count = 0

    print(f"\nGenerating AI counterparts...")
    for i, article in enumerate(tqdm(articles, desc="Processing")):
        try:
            # Extract text based on structure
            if isinstance(article, dict):
                original_text = article.get('text', article.get('article', article.get('document', '')))
                source = article.get('source', 'unknown')
                category = article.get('category', article.get('label_name', 'news'))
            else:
                original_text = str(article)
                source = 'unknown'
                category = 'news'

            if not original_text or len(original_text) < 50:
                continue

            # Generate AI version
            ai_text = generator.generate_article(original_text)

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
            else:
                failed_count += 1

            # Save periodically
            if (i + 1) % 10 == 0:
                save_results(ai_articles, output_path, model_name)

            # Rate limiting for APIs
            if model_name in ['gpt-3.5', 'gpt-4', 'claude-3', 'hf-mistral', 'hf-llama']:
                time.sleep(0.5)  # Avoid rate limits

        except Exception as e:
            print(f"\nError processing article {i}: {e}")
            failed_count += 1
            continue

    # Final save
    save_results(ai_articles, output_path, model_name)

    print(f"\n" + "=" * 60)
    print(f"GENERATION COMPLETE")
    print(f"=" * 60)
    print(f"Successfully generated: {len(ai_articles)} articles")
    print(f"Failed: {failed_count} articles")
    print(f"Output directory: {output_path}")


def save_results(ai_articles: List[Dict], output_path: Path, model_name: str):
    """Save AI-generated articles to files"""
    if not ai_articles:
        return

    # Save as JSON
    json_file = output_path / f"ai_articles_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(ai_articles, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        'model': model_name,
        'total_articles': len(ai_articles),
        'generation_date': datetime.now().isoformat(),
        'avg_length': sum(a['ai_length'] for a in ai_articles) / len(ai_articles),
        'categories': {}
    }

    for article in ai_articles:
        cat = article.get('category', 'unknown')
        summary['categories'][cat] = summary['categories'].get(cat, 0) + 1

    summary_file = output_path / f"summary_{model_name}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(ai_articles)} articles to {json_file}")


def list_available_models():
    """List all available models with details"""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS FOR AI ARTICLE GENERATION")
    print("=" * 80)

    # Create a dummy instance just to get configs without initializing a model
    configs = {
        # Open-source models (no auth required)
        'phi-2': {'model_id': 'microsoft/phi-2', 'type': 'local', 'description': 'Small but capable, low resource, no auth needed', 'vram_required': '6GB', 'quality': 'good'},
        'gpt2-large': {'model_id': 'gpt2-large', 'type': 'local', 'description': 'Classic GPT-2, no auth required', 'vram_required': '4GB', 'quality': 'good'},
        'gpt2-xl': {'model_id': 'gpt2-xl', 'type': 'local', 'description': 'Larger GPT-2, better quality', 'vram_required': '8GB', 'quality': 'very good'},
        'opt-1.3b': {'model_id': 'facebook/opt-1.3b', 'type': 'local', 'description': 'Meta OPT model, good quality', 'vram_required': '4GB', 'quality': 'good'},
        'opt-2.7b': {'model_id': 'facebook/opt-2.7b', 'type': 'local', 'description': 'Larger OPT, better performance', 'vram_required': '8GB', 'quality': 'very good'},
        # HF Inference API
        'hf-mistral': {'type': 'hf-inference', 'description': 'Via HF Inference API (free tier)', 'quality': 'excellent'},
        # Commercial
        'gpt-3.5': {'type': 'openai', 'description': 'OpenAI GPT-3.5', 'quality': 'very good'},
        'gpt-4': {'type': 'openai', 'description': 'OpenAI GPT-4', 'quality': 'state-of-the-art'},
    }

    print("\n🌟 RECOMMENDED OPEN-SOURCE MODELS (No Auth Required):")
    print("-" * 50)
    for name, config in configs.items():
        if config['type'] == 'local' and 'auth' not in config.get('description', '').lower():
            print(f"\n{name}:")
            print(f"  • Description: {config['description']}")
            print(f"  • Quality: {config['quality']}")
            if 'vram_required' in config:
                print(f"  • VRAM Required: {config['vram_required']}")

    print("\n\n💰 COMMERCIAL MODELS (API Key Required):")
    print("-" * 50)
    for name, config in configs.items():
        if config['type'] in ['openai', 'anthropic']:
            print(f"\n{name}:")
            print(f"  • Description: {config['description']}")
            print(f"  • Quality: {config['quality']}")

    print("\n\n📌 RECOMMENDATIONS:")
    print("-" * 50)
    print("""
    1. For LOCAL TESTING (No API costs):
       • mistral-7b: Best quality/speed balance (16GB VRAM)
       • zephyr-7b: Excellent quality, based on Mistral (16GB VRAM)
       • phi-2: Low resource usage (6GB VRAM, works on CPU)

    2. For FREE CLOUD (Limited rate):
       • hf-mistral: Via Hugging Face API (free tier)
       • hf-llama: Via Hugging Face API (free tier)

    3. For BEST QUALITY (Paid):
       • gpt-4: OpenAI's best
       • claude-3: Anthropic's Claude 3
       • gpt-3.5: Good quality, lower cost
    """)


def main():
    parser = argparse.ArgumentParser(description='Generate AI counterparts for news articles')

    parser.add_argument('--input', '-i', type=str,
                       default='../../data/datasets/combined_human_news_full.json',
                       help='Input JSON file with human articles')

    parser.add_argument('--output', '-o', type=str,
                       default='../../data/ai_generated',
                       help='Output directory for AI articles')

    parser.add_argument('--model', '-m', type=str,
                       default='mistral-7b',
                       help='Model to use (see --list-models)')

    parser.add_argument('--limit', '-l', type=int,
                       help='Limit number of articles to process')

    parser.add_argument('--api-key', '-k', type=str,
                       help='API key for commercial models')

    parser.add_argument('--list-models', action='store_true',
                       help='List all available models')

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
    else:
        process_dataset(
            args.input,
            args.output,
            args.model,
            args.limit,
            args.api_key
        )


if __name__ == "__main__":
    main()
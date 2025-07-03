#!/usr/bin/env python3
"""
Translation Dataset Filter with GPT Embeddings

This script filters translation datasets based on length and ratio criteria.
It processes JSON files containing translation pairs and outputs filtered datasets.
Now supports both sentence-transformers and OpenAI GPT embeddings for semantic alignment.

Usage:
    python filter_translation_dataset.py [OPTIONS]

Options:
    -i, --input-dir     Input directory containing dataset folders (default: current directory)
    -o, --output-dir    Output directory for filtered datasets (default: ./filtered)
    --min-length        Minimum token length (default: 3)
    --max-length        Maximum token length (default: 200)
    --max-ratio         Maximum length ratio between source/target (default: 1.5)
    --min-ratio         Minimum length ratio between source/target (default: 0.67)
    --max-diff          Maximum token difference between source/target (default: 50)
    --enable-alignment  Enable semantic alignment checking using sentence embeddings
    --embedding-model   Embedding model to use (default: all-MiniLM-L6-v2)
    --use-gpt           Use OpenAI GPT embeddings instead of sentence-transformers
    --gpt-model         GPT embedding model (default: text-embedding-3-large)
    --openai-token      OpenAI API token (can also be set via OPENAI_TOKEN env var)
    --min-similarity    Minimum cosine similarity for alignment (default: 0.5)
    --batch-size        Batch size for embedding computation (default: 32)
    -h, --help          Show this help message

Expected JSON format:
    {"source": "source text", "target": "target text"}
    
The script supports both JSONL (one JSON object per line) and JSON array formats.
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from tqdm import tqdm
from dotenv import load_dotenv
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables to store embedding models
_embedding_model = None
_gpt_embeddings = None


def get_openai_token(provided_token: Optional[str] = None) -> Optional[str]:
    """Get OpenAI token from environment or user input."""
    # Try provided token first
    if provided_token:
        return provided_token
    
    # Try environment variable
    token = os.getenv("OPENAI_TOKEN") or os.getenv("OPENAI_API_KEY")
    if token:
        return token
    
    # Ask user for token
    print("\nOpenAI API token not found in environment variables (OPENAI_TOKEN or OPENAI_API_KEY).")
    print("Please provide your OpenAI API token:")
    token = input("OpenAI Token: ").strip()
    
    if not token:
        print("No token provided. Cannot use GPT embeddings.")
        return None
    
    return token


def get_gpt_embedding_model(model_name: str = "text-embedding-3-large", 
                           api_token: Optional[str] = None):
    """Initialize and return the GPT embedding model."""
    global _gpt_embeddings
    if _gpt_embeddings is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            
            token = get_openai_token(api_token)
            if not token:
                return None
            
            print(f"Initializing OpenAI embeddings with model: {model_name}")
            _gpt_embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=token
            )
            print("✓ OpenAI embeddings initialized successfully")
        except ImportError:
            print("Error: langchain-openai not installed. Install with: pip install langchain-openai")
            return None
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {e}")
            return None
    return _gpt_embeddings


def get_sentence_transformer_model(model_name: str):
    """Initialize and return the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence transformer model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            print("✓ Sentence transformer model loaded successfully")
        except ImportError:
            print("Error: sentence-transformers not installed. Install with: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            return None
    return _embedding_model


def compute_gpt_embeddings(texts: List[str], model, batch_size: int = 32) -> Optional[np.ndarray]:
    """Compute GPT embeddings for a list of texts with progress bar."""
    if model is None:
        return None
    
    try:
        print(f"Generating GPT embeddings for {len(texts)} texts...")
        
        # Process in batches to avoid rate limits and memory issues
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i:i+batch_size]
            try:
                batch_embeddings = model.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Return None if any batch fails
                return None
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error computing GPT embeddings: {e}")
        return None


def compute_sentence_transformer_embeddings(texts: List[str], model, batch_size: int = 32) -> Optional[np.ndarray]:
    """Compute sentence transformer embeddings for a list of texts."""
    if model is None:
        return None
    
    try:
        print(f"Computing sentence transformer embeddings for {len(texts)} texts...")
        
        # Process in batches to avoid memory issues
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)
    except Exception as e:
        print(f"Error computing sentence transformer embeddings: {e}")
        return None


def check_alignment(source_texts: List[str], target_texts: List[str], 
                   use_gpt: bool = False,
                   gpt_model: str = "text-embedding-3-large",
                   sentence_model: str = "all-MiniLM-L6-v2",
                   min_similarity: float = 0.5, 
                   batch_size: int = 32,
                   openai_token: Optional[str] = None) -> Optional[List[float]]:
    """Check semantic alignment between source and target texts."""
    
    if use_gpt:
        model = get_gpt_embedding_model(gpt_model, openai_token)
        if model is None:
            return None
        
        # Compute embeddings using GPT
        source_embeddings = compute_gpt_embeddings(source_texts, model, batch_size)
        target_embeddings = compute_gpt_embeddings(target_texts, model, batch_size)
    else:
        model = get_sentence_transformer_model(sentence_model)
        if model is None:
            return None
        
        # Compute embeddings using sentence transformers
        source_embeddings = compute_sentence_transformer_embeddings(source_texts, model, batch_size)
        target_embeddings = compute_sentence_transformer_embeddings(target_texts, model, batch_size)
    
    if source_embeddings is None or target_embeddings is None:
        return None
    
    # Compute cosine similarities
    print("Computing cosine similarities...")
    similarities = []
    for i in tqdm(range(len(source_embeddings)), desc="Computing similarities"):
        sim = cosine_similarity([source_embeddings[i]], [target_embeddings[i]])[0][0]
        similarities.append(float(sim))
    
    return similarities


def load_json_dataset(filepath: str) -> List[Dict[str, str]]:
    """Load JSON dataset from file, supporting both JSONL and JSON array formats."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping...")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == '[':
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        return [entry for entry in data if 'source' in entry and 'target' in entry]
                    else:
                        print(f"Error: Expected JSON array in {filepath}")
                        return []
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON array in {filepath}: {e}")
                    return []
            else:
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if 'source' not in entry or 'target' not in entry:
                            print(f"Warning: Missing source/target in line {line_num}")
                            continue
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {filepath}: {e}")
                        continue
                return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def filter_dataset(data: List[Dict[str, str]], 
                  min_length: int = 3, 
                  max_length: int = 200, 
                  max_ratio: float = 3.0, 
                  min_ratio: float = 0.33, 
                  max_diff: int = 50,
                  enable_alignment: bool = False,
                  use_gpt: bool = False,
                  gpt_model: str = "text-embedding-3-large",
                  sentence_model: str = "all-MiniLM-L6-v2",
                  min_similarity: float = 0.5,
                  batch_size: int = 32,
                  openai_token: Optional[str] = None) -> Tuple[List[Dict], List[Dict], Dict]:
    """Filter dataset based on length and ratio criteria."""
    valid_entries = []
    invalid_entries = []
    stats = {
        'total': 0,
        'too_short': 0,
        'too_long': 0,
        'bad_ratio': 0,
        'large_diff': 0,
        'empty': 0,
        'poor_alignment': 0,
        'valid': 0
    }
    
    for entry in data:
        stats['total'] += 1
        
        src = entry.get('source', '').strip()
        tgt = entry.get('target', '').strip()
        
        if not src or not tgt:
            invalid_entry = entry.copy()
            invalid_entry['filter_reason'] = "Empty source or target"
            invalid_entries.append(invalid_entry)
            stats['empty'] += 1
            continue
        
        src_tokens = src.split()
        tgt_tokens = tgt.split()
        
        src_len = len(src_tokens)
        tgt_len = len(tgt_tokens)
        
        if src_len < min_length or tgt_len < min_length:
            invalid_entry = entry.copy()
            invalid_entry['filter_reason'] = f"Too short: {src_len}/{tgt_len} tokens"
            invalid_entries.append(invalid_entry)
            stats['too_short'] += 1
            continue
            
        if src_len > max_length or tgt_len > max_length:
            invalid_entry = entry.copy()
            invalid_entry['filter_reason'] = f"Too long: {src_len}/{tgt_len} tokens"
            invalid_entries.append(invalid_entry)
            stats['too_long'] += 1
            continue
        
        ratio = max(src_len, tgt_len) / min(src_len, tgt_len) if min(src_len, tgt_len) > 0 else float('inf')
            
        if ratio > max_ratio:
            invalid_entry = entry.copy()
            invalid_entry['filter_reason'] = f"Bad ratio: {ratio:.2f} ({src_len}/{tgt_len})"
            invalid_entries.append(invalid_entry)
            stats['bad_ratio'] += 1
            continue
        
        diff = abs(src_len - tgt_len)
        if diff > max_diff:
            invalid_entry = entry.copy()
            invalid_entry['filter_reason'] = f"Large diff: {diff} tokens ({src_len}/{tgt_len})"
            invalid_entries.append(invalid_entry)
            stats['large_diff'] += 1
            continue
        
        valid_entry = entry.copy()
        valid_entry['src_len'] = src_len
        valid_entry['tgt_len'] = tgt_len
        valid_entry['length_ratio'] = ratio
        valid_entries.append(valid_entry)
        stats['valid'] += 1

    # Check semantic alignment if enabled
    if enable_alignment and valid_entries:
        embedding_type = "GPT" if use_gpt else "Sentence Transformer"
        print(f"Checking semantic alignment using {embedding_type} for {len(valid_entries)} valid entries...")
        
        source_texts = [entry['source'] for entry in valid_entries]
        target_texts = [entry['target'] for entry in valid_entries]
        
        similarities = check_alignment(source_texts, target_texts, 
                                     use_gpt=use_gpt,
                                     gpt_model=gpt_model,
                                     sentence_model=sentence_model,
                                     min_similarity=min_similarity,
                                     batch_size=batch_size,
                                     openai_token=openai_token)
        
        if similarities is not None:
            # Re-filter based on alignment
            alignment_valid = []
            alignment_invalid = []
            
            for i, (entry, sim) in enumerate(zip(valid_entries, similarities)):
                entry_with_sim = entry.copy()
                entry_with_sim['similarity'] = sim
                
                if sim >= min_similarity:
                    alignment_valid.append(entry_with_sim)
                else:
                    entry_with_sim['filter_reason'] = f"Poor alignment: {sim:.3f} < {min_similarity}"
                    alignment_invalid.append(entry_with_sim)
                    invalid_entries.append(entry_with_sim)
                    stats['poor_alignment'] += 1
                    stats['valid'] -= 1
            
            valid_entries = alignment_valid
            print(f"✓ Alignment check complete. {len(alignment_valid)} entries passed similarity threshold")
        else:
            print("⚠ Alignment check failed, proceeding without semantic filtering")

    return valid_entries, invalid_entries, stats


def save_filtered_dataset(entries: List[Dict], output_path: str):
    """Save filtered dataset to JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def print_statistics(split_name: str, stats: Dict):
    """Print filtering statistics for a dataset split."""
    if stats['total'] == 0:
        print(f"\n=== {split_name.upper()} (Empty) ===")
        return
        
    print(f"\n=== {split_name.upper()} Statistics ===")
    print(f"Total pairs: {stats['total']}")
    print(f"Valid pairs: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Empty: {stats['empty']}")
    print(f"Too short: {stats['too_short']}")
    print(f"Too long: {stats['too_long']}")
    print(f"Bad ratio: {stats['bad_ratio']}")
    print(f"Large diff: {stats['large_diff']}")
    if stats.get('poor_alignment', 0) > 0:
        print(f"Poor alignment: {stats['poor_alignment']}")


def analyze_data(input_dir: str, output_dir: str, filter_params: Dict):
    """Analyze and filter all JSON files in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    total_stats = {'total': 0, 'valid': 0, 'invalid': 0}

    for fname in os.listdir(input_dir):
        if not fname.endswith('.json') or fname.startswith('.'):
            continue

        split = os.path.splitext(fname)[0]
        input_file = os.path.join(input_dir, fname)

        data = load_json_dataset(input_file)
        if not data:
            continue

        print(f"\nProcessing {split} split...")

        valid, invalid, stats = filter_dataset(data, **filter_params)

        if valid:
            output_file = os.path.join(output_dir, f"{split}_filtered.json")
            save_filtered_dataset(valid, output_file)
            print(f"Saved {len(valid)} valid entries to {output_file}")

        if invalid:
            invalid_file = os.path.join(output_dir, f"{split}_invalid.json")
            save_filtered_dataset(invalid, invalid_file)
            print(f"Saved {len(invalid)} invalid entries to {invalid_file}")

        print_statistics(split, stats)

        total_stats['total'] += stats['total']
        total_stats['valid'] += stats['valid']
        total_stats['invalid'] += stats['total'] - stats['valid']

    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total pairs processed: {total_stats['total']}")
    if total_stats['total'] == 0:
        return
    print(f"Valid pairs: {total_stats['valid']} ({total_stats['valid']/total_stats['total']*100:.1f}%)")
    print(f"Invalid pairs: {total_stats['invalid']} ({total_stats['invalid']/total_stats['total']*100:.1f}%)")


def main():
    """Main function to parse arguments and run the filtering process."""
    parser = argparse.ArgumentParser(
        description="Filter translation datasets based on length and ratio criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter datasets in current directory with default parameters
    python filter_translation_dataset.py

    # Filter specific input directory with custom output
    python filter_translation_dataset.py -i /path/to/datasets -o /path/to/filtered

    # Custom filtering parameters
    python filter_translation_dataset.py --min-length 5 --max-length 100 --max-ratio 2.0
    
    # Enable semantic alignment checking with sentence transformers
    python filter_translation_dataset.py --enable-alignment --min-similarity 0.6
    
    # Enable semantic alignment checking with GPT embeddings
    python filter_translation_dataset.py --enable-alignment --use-gpt --gpt-model text-embedding-3-large
    
    # Use GPT embeddings with custom token
    python filter_translation_dataset.py --enable-alignment --use-gpt --openai-token your_token_here
        """
    )
    
    parser.add_argument('-i', '--input-dir', 
                       default='.', 
                       help='Input directory containing dataset folders (default: current directory)')
    parser.add_argument('-o', '--output-dir', 
                       default='./filtered', 
                       help='Output directory for filtered datasets (default: ./filtered)')
    parser.add_argument('--min-length', 
                       type=int, 
                       default=3, 
                       help='Minimum token length (default: 3)')
    parser.add_argument('--max-length', 
                       type=int, 
                       default=200, 
                       help='Maximum token length (default: 200)')
    parser.add_argument('--max-ratio', 
                       type=float, 
                       default=1.5, 
                       help='Maximum length ratio between source/target (default: 1.5)')
    parser.add_argument('--min-ratio', 
                       type=float, 
                       default=0.67, 
                       help='Minimum length ratio between source/target (default: 0.67)')
    parser.add_argument('--max-diff', 
                       type=int, 
                       default=50, 
                       help='Maximum token difference between source/target (default: 50)')
    parser.add_argument('--enable-alignment', 
                       action='store_true',
                       help='Enable semantic alignment checking using sentence embeddings')
    parser.add_argument('--use-gpt', 
                       action='store_true',
                       help='Use OpenAI GPT embeddings instead of sentence-transformers')
    parser.add_argument('--gpt-model', 
                       default='text-embedding-3-large',
                       help='GPT embedding model (default: text-embedding-3-large)')
    parser.add_argument('--embedding-model', 
                       default='all-MiniLM-L6-v2',
                       help='Sentence transformer model to use (default: all-MiniLM-L6-v2)')
    parser.add_argument('--openai-token', 
                       help='OpenAI API token (can also be set via OPENAI_TOKEN env var)')
    parser.add_argument('--min-similarity', 
                       type=float, 
                       default=0.5,
                       help='Minimum cosine similarity for alignment (default: 0.5)')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=32,
                       help='Batch size for embedding computation (default: 32)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_gpt and not args.enable_alignment:
        print("Warning: --use-gpt flag requires --enable-alignment to be set")
        args.enable_alignment = True
    
    filter_params = {
        'min_length': args.min_length,
        'max_length': args.max_length,
        'max_ratio': args.max_ratio,
        'min_ratio': args.min_ratio,
        'max_diff': args.max_diff,
        'enable_alignment': args.enable_alignment,
        'use_gpt': args.use_gpt,
        'gpt_model': args.gpt_model,
        'sentence_model': args.embedding_model,
        'min_similarity': args.min_similarity,
        'batch_size': args.batch_size,
        'openai_token': args.openai_token
    }
    
    data_path = args.input_dir
    
    if not os.path.exists(data_path):
        print(f"Error: Input directory '{data_path}' does not exist")
        return
    
    print(f"Processing datasets in: {data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Filter parameters: {filter_params}")
    
    # Process each subdirectory containing datasets
    for data in os.listdir(data_path):
        full_path = os.path.join(data_path, data)
        if os.path.isdir(full_path) and not data.startswith('.'):
            print(f"\n========================\nProcessing dataset: {data}\n========================")
            input_dir = full_path
            output_dir = os.path.join(args.output_dir, data)
            
            analyze_data(input_dir, output_dir, filter_params)


if __name__ == "__main__":
    main()
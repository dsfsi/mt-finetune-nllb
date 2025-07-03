#!/usr/bin/env python3
"""
Translation Dataset Filter

This script filters translation datasets based on length and ratio criteria.
It processes JSON files containing translation pairs and outputs filtered datasets.

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
    -h, --help          Show this help message

Expected JSON format:
    {"source": "source text", "target": "target text"}
    
The script supports both JSONL (one JSON object per line) and JSON array formats.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple


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
                  max_diff: int = 50) -> Tuple[List[Dict], List[Dict], Dict]:
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

        print(f"\n{split} split...")

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
    
    args = parser.parse_args()
    
    # Convert min_ratio to max_ratio format for consistency
    filter_params = {
        'min_length': args.min_length,
        'max_length': args.max_length,
        'max_ratio': args.max_ratio,
        'min_ratio': args.min_ratio,
        'max_diff': args.max_diff
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
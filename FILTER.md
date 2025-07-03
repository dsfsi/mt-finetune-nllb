# Translation Dataset Filter with Semantic Alignment

A comprehensive Python script for filtering translation datasets based on length, ratio, and semantic alignment criteria. The script processes JSON files containing translation pairs and outputs filtered datasets with detailed statistics and filtering reasons.

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

OR install dependencies individually:

```bash
# Core dependencies
pip install numpy scikit-learn tqdm python-dotenv

# For semantic alignment with sentence transformers
pip install sentence-transformers

# For OpenAI GPT embeddings
pip install langchain-openai

# Optional: For enhanced progress tracking
pip install tqdm
```

## Quick Start

### Basic Usage

```bash
# Filter datasets in current directory with default parameters
python filter_dataset.py

# Filter specific directory
python filter_dataset.py -i /path/to/datasets -o /path/to/filtered

# Custom filtering parameters
python filter_dataset.py --min-length 5 --max-length 100 --max-ratio 2.0
```

### With Semantic Alignment

```bash
# Enable semantic alignment with sentence transformers
python filter_dataset.py --enable-alignment --min-similarity 0.6

# Use OpenAI GPT embeddings for semantic alignment
python filter_dataset.py --enable-alignment --use-gpt --openai-token your_token_here

# Custom embedding model
python filter_dataset.py --enable-alignment --embedding-model all-mpnet-base-v2
```

## Command Line Arguments

### Input/Output Options

- `-i, --input-dir`: Input directory containing dataset folders (default: current directory)
- `-o, --output-dir`: Output directory for filtered datasets (default: ./filtered)

### Length Filtering Parameters

- `--min-length`: Minimum token length (default: 3)
- `--max-length`: Maximum token length (default: 200)
- `--max-ratio`: Maximum length ratio between source/target (default: 1.5)
- `--min-ratio`: Minimum length ratio between source/target (default: 0.67)
- `--max-diff`: Maximum token difference between source/target (default: 50)

### Semantic Alignment Options

- `--enable-alignment`: Enable semantic alignment checking
- `--use-gpt`: Use OpenAI GPT embeddings instead of sentence transformers
- `--gpt-model`: GPT embedding model (default: text-embedding-3-large)
- `--embedding-model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `--openai-token`: OpenAI API token (can also be set via OPENAI_TOKEN env var)
- `--min-similarity`: Minimum cosine similarity for alignment (default: 0.5)
- `--batch-size`: Batch size for embedding computation (default: 32)

## Data Format

### Expected Input Format

The script expects JSON files containing translation pairs with the following structure:

```json
{"source": "Hello world", "target": "Hola mundo"}
```

### Supported Formats

**JSONL Format (one JSON object per line):**
```jsonl
{"source": "Hello world", "target": "Hola mundo"}
{"source": "How are you?", "target": "¿Cómo estás?"}
{"source": "Good morning", "target": "Buenos días"}
```

**JSON Array Format:**
```json
[
    {"source": "Hello world", "target": "Hola mundo"},
    {"source": "How are you?", "target": "¿Cómo estás?"},
    {"source": "Good morning", "target": "Buenos días"}
]
```

### Directory Structure

The script processes datasets organized in subdirectories:

```
input_directory/
├── dataset1/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── dataset2/
│   ├── train.json
│   └── test.json
└── dataset3/
    └── data.json
```

## Output Files

### Filtered Datasets

- `*_filtered.json`: Valid translation pairs that passed all filters
- `*_invalid.json`: Rejected pairs with detailed filtering reasons

### Output Structure

**Filtered entries include additional metadata:**
```json
{
    "source": "Hello world",
    "target": "Hola mundo",
    "src_len": 2,
    "tgt_len": 2,
    "length_ratio": 1.0,
    "similarity": 0.85
}
```

**Invalid entries include rejection reasons:**
```json
{
    "source": "Hi",
    "target": "This is a very long translation that doesn't match the source",
    "filter_reason": "Bad ratio: 8.00 (1/8)"
}
```

## Filtering Criteria

### Length-based Filters

1. **Minimum Length**: Both source and target must have at least `min_length` tokens
2. **Maximum Length**: Both source and target must have at most `max_length` tokens
3. **Length Ratio**: The ratio between longer and shorter text must be within bounds
4. **Length Difference**: Absolute difference in token count must not exceed `max_diff`

### Semantic Alignment (Optional)

When enabled, the script computes semantic similarity between source and target texts using:

- **Sentence Transformers**: Local models for fast, offline processing
- **OpenAI GPT Embeddings**: Cloud-based embeddings for potentially better quality

The cosine similarity between embeddings must exceed `min_similarity` threshold.

## Environment Variables

### OpenAI Configuration

Set your OpenAI API token using environment variables:

```bash
# Option 1: OPENAI_TOKEN
export OPENAI_TOKEN="your-api-token-here"

# Option 2: OPENAI_API_KEY
export OPENAI_API_KEY="your-api-token-here"
```

Or create a `.env` file:
```
OPENAI_TOKEN=your-api-token-here
```

## Examples

### Example 1: Basic Filtering

```bash
# Filter with conservative parameters
python filter_dataset.py \
    --min-length 5 \
    --max-length 50 \
    --max-ratio 1.2 \
    --max-diff 10
```

### Example 2: Semantic Alignment with Sentence Transformers

```bash
# Use high-quality sentence transformer model
python filter_dataset.py \
    --enable-alignment \
    --embedding-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --min-similarity 0.7 \
    --batch-size 16
```

### Example 3: OpenAI GPT Embeddings

```bash
# Use OpenAI embeddings for semantic filtering
python filter_dataset.py \
    --enable-alignment \
    --use-gpt \
    --gpt-model text-embedding-3-large \
    --min-similarity 0.6 \
    --batch-size 8
```

### Example 4: Processing Large Datasets

```bash
# Optimize for large datasets
python filter_dataset.py \
    -i /path/to/large/datasets \
    -o /path/to/filtered/output \
    --enable-alignment \
    --batch-size 64 \
    --min-length 3 \
    --max-length 200
```

## Statistics and Reporting

The script provides comprehensive statistics for each dataset:

```
=== TRAIN Statistics ===
Total pairs: 10000
Valid pairs: 8500 (85.0%)
Empty: 50
Too short: 200
Too long: 150
Bad ratio: 800
Large diff: 300
Poor alignment: 0

=== OVERALL STATISTICS ===
Total pairs processed: 25000
Valid pairs: 21000 (84.0%)
Invalid pairs: 4000 (16.0%)
```

## Embedding Models

### Sentence Transformer Models

Popular models for multilingual tasks:

- `all-MiniLM-L6-v2`: Fast, lightweight, good for most tasks
- `all-mpnet-base-v2`: Higher quality, slower
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`: Multilingual support
- `sentence-transformers/LaBSE`: Language-agnostic BERT sentence embedding

### OpenAI GPT Models

Available embedding models:

- `text-embedding-3-large`: Latest, highest quality (3072 dimensions)
- `text-embedding-3-small`: Faster, smaller (1536 dimensions)
- `text-embedding-ada-002`: Legacy model, still effective

## Performance Optimization

### Memory Management

- **Batch Processing**: Processes embeddings in configurable batches
- **Progress Tracking**: Shows progress for large datasets
- **Error Recovery**: Continues processing even if individual batches fail

### Speed Optimization

```bash
# For faster processing with sentence transformers
python filter_dataset.py \
    --enable-alignment \
    --embedding-model all-MiniLM-L6-v2 \
    --batch-size 64

# For better quality with OpenAI (slower)
python filter_dataset.py \
    --enable-alignment \
    --use-gpt \
    --gpt-model text-embedding-3-large \
    --batch-size 8
```

## Troubleshooting

### Common Issues

**OpenAI API Issues:**
```bash
# Check your token
export OPENAI_TOKEN="your-token-here"

# Reduce batch size to avoid rate limits
--batch-size 4
```

**Memory Issues:**
```bash
# Reduce batch size
--batch-size 16

# Use smaller embedding model
--embedding-model all-MiniLM-L6-v2
```

**Large Dataset Processing:**
```bash
# Process in smaller batches
--batch-size 32

# Disable alignment for initial filtering
# Then enable for final filtering
```

### Error Messages

- **"OpenAI API token not found"**: Set OPENAI_TOKEN environment variable
- **"sentence-transformers not installed"**: Install with `pip install sentence-transformers`
- **"langchain-openai not installed"**: Install with `pip install langchain-openai`

## Best Practices

### Dataset Preparation

1. **Clean Data**: Remove obviously malformed entries before filtering
2. **Consistent Formatting**: Ensure all files use the same JSON structure
3. **Language Consistency**: Keep source and target languages consistent within datasets

### Parameter Tuning

1. **Start Conservative**: Begin with strict parameters and gradually relax
2. **Monitor Statistics**: Use output statistics to guide parameter adjustment
3. **Validate Results**: Manually check a sample of filtered results

### Semantic Alignment

1. **Model Selection**: Choose embedding models appropriate for your languages
2. **Similarity Thresholds**: Start with 0.5 and adjust based on results
3. **Batch Size**: Balance speed vs. memory usage based on your hardware
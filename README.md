# NLLB Fine-tuning and Evaluation Script

A comprehensive Python script for fine-tuning NLLB-200 (No Language Left Behind) models on custom translation datasets with support for new language pairs, vocabulary extension, and multiple evaluation metrics.

## Features

- 🌍 **Multi-language Support**: Fine-tune on any language pair supported by NLLB-200
- 📚 **New Language Addition**: Extend vocabulary with new language tokens
- 📊 **Flexible Data Formats**: Support for JSON, JSONL, CSV, and TSV files
- 🔄 **Bidirectional Training**: Automatic bidirectional translation training
- 📈 **Smart Data Splitting**: Automatic train/dev/test splits or use pre-split data
- 🎯 **Prediction Pipeline**: Built-in evaluation on test sets
- 🛠️ **Memory Optimization**: GPU memory management and cleanup
- 📝 **Comprehensive Logging**: Detailed training progress and model information
- 📊 **Evaluation Metrics**: Supports exact match, BLEU, chrF, chrF++, TER, COMET, and length statistics
- 🔄 **Interactive Translation**: Real-time translation testing with user input

## Installation

### Requirements

```bash
pip install torch transformers datasets
pip install pandas numpy scikit-learn tqdm
pip install sacremoses unicodedata2
```

### Optional Dependencies

For enhanced evaluation metrics:

```bash
# For BLEU, chrF, TER metrics
pip install sacrebleu

# For COMET metric (neural evaluation)
pip install unbabel-comet
```

## Usage

### Basic Training

```bash
python finetune.py \
    --do_train \
    --data_path data/translation_pairs.json \
    --src_lang eng_Latn \
    --tgt_lang fra_Latn \
    --src_col source \
    --tgt_col target \
    --output_dir ./models/eng-fra \
    --epochs 3 \
    --batch_size 8
```

### Training with New Language

```bash
python finetune.py \
    --do_train \
    --data_path data/eng_newlang.json \
    --src_lang eng_Latn \
    --tgt_lang new_Latn \
    --new_lang new_Latn \
    --similar_lang fra_Latn \
    --output_dir ./models/eng-newlang \
    --epochs 5
```

### Evaluation Only

```bash
python finetune.py \
    --do_eval \
    --model_path ./models/eng-fra \
    --test_file data/test.json \
    --src_lang eng_Latn \
    --tgt_lang fra_Latn \
    --metrics bleu chrf comet \
    --output_dir ./results
```

### Interactive Translation

```bash
python finetune.py \
    --interactive \
    --model_path ./models/eng-fra \
    --src_lang eng_Latn \
    --tgt_lang fra_Latn
```

## Data Format

### JSON/JSONL Format

**Single JSON file:**
```json
[
    {"source": "Hello world", "target": "Bonjour le monde"},
    {"source": "How are you?", "target": "Comment allez-vous?"}
]
```

**JSONL format (one JSON object per line):**
```jsonl
{"source": "Hello world", "target": "Bonjour le monde"}
{"source": "How are you?", "target": "Comment allez-vous?"}
```

**With predefined splits:**
```json
[
    {"source": "Hello", "target": "Bonjour", "split": "train"},
    {"source": "World", "target": "Monde", "split": "test"}
]
```

### CSV/TSV Format

```csv
source,target,split
"Hello world","Bonjour le monde",train
"How are you?","Comment allez-vous?",test
```

## Command Line Arguments

### Model Arguments

- `--model_name`: Base NLLB model (default: `facebook/nllb-200-distilled-600M`)
- `--model_path`: Path to existing fine-tuned model
- `--output_dir`: Output directory for models and results

### Data Arguments

- `--data_path`: Single data file path
- `--train_file`: Separate training data file
- `--dev_file`: Development/validation data file  
- `--test_file`: Test data file
- `--src_col`: Source text column name (default: "source")
- `--tgt_col`: Target text column name (default: "target")
- `--src_lang`: Source language code (e.g., "eng_Latn")
- `--tgt_lang`: Target language code (e.g., "fra_Latn")

### Language Extension

- `--new_lang`: New language code to add to tokenizer
- `--similar_lang`: Similar language for embedding initialization

### Training Arguments

- `--do_train`: Enable training mode
- `--do_eval`: Enable evaluation mode
- `--do_predict`: Enable prediction mode
- `--interactive`: Enable interactive translation mode
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_length`: Maximum sequence length (default: 512)

### Data Splitting

- `--test_size`: Test set fraction (default: 0.1)
- `--dev_size`: Development set fraction (default: 0.1)

### Evaluation Metrics

- `--metrics`: Metrics to compute (choices: exact_match, bleu, chrf, chrfpp, ter, comet, length_stats)
- `--comet_model`: COMET model for neural evaluation (e.g., "Unbabel/wmt22-comet-da")

## Language Codes

NLLB uses specific language codes. Common examples:

- English: `eng_Latn`
- Hausa: `hau_Latn`
- Zulu: `zul_Latn`
- Sepedi: `nso_Latn`
- Amharic: `amh_Ethi`
- Arabic: `ara_Arab`
- Hindi: `hin_Deva`

[Full list of supported languages](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/README.md)

## Output Files

The script generates several output files:

### Model Files
- `pytorch_model.bin`: Fine-tuned model weights
- `config.json`: Model configuration
- `tokenizer.json`: Tokenizer configuration
- `training_info.json`: Training statistics and arguments

### Evaluation Files
- `{dataset}_results.json`: Complete evaluation results
- `{dataset}_predictions.txt`: Human-readable predictions
- `{dataset}_metrics.json`: Metrics summary

## Examples

### Example 1: Fine-tune for English-French

```bash
# Prepare data in JSON format
echo '[
    {"source": "Hello", "target": "Bonjour"},
    {"source": "Goodbye", "target": "Au revoir"},
    {"source": "Thank you", "target": "Merci"}
]' > eng_fra_data.json

# Train the model
python finetune.py \
    --do_train \
    --data_path eng_fra_data.json \
    --src_lang eng_Latn \
    --tgt_lang fra_Latn \
    --output_dir ./models/eng-fra \
    --epochs 2 \
    --batch_size 4
```

### Example 2: Add New Language

```bash
# Train with new language token
python finetune.py \
    --do_train \
    --data_path new_language_data.json \
    --src_lang eng_Latn \
    --tgt_lang xyz_Latn \
    --new_lang xyz_Latn \
    --similar_lang fra_Latn \
    --output_dir ./models/eng-xyz \
    --epochs 5
```

### Example 3: Comprehensive Evaluation

```bash
# Evaluate with multiple metrics
python finetune.py \
    --do_eval \
    --model_path ./models/eng-fra \
    --test_file test_data.json \
    --src_lang eng_Latn \
    --tgt_lang fra_Latn \
    --metrics exact_match bleu chrf comet \
    --comet_model Unbabel/wmt22-comet-da \
    --output_dir ./evaluation_results
```

## Evaluation Metrics

### Available Metrics

1. **Exact Match**: Exact string matching accuracy
2. **BLEU**: Bilingual Evaluation Understudy score
3. **chrF**: Character-level F-score
4. **chrF++**: Enhanced character-level F-score with word order
5. **TER**: Translation Error Rate
6. **COMET**: Neural evaluation metric using pretrained models
7. **Length Stats**: Statistical analysis of translation lengths

### COMET Models

Popular COMET models for evaluation:
- `Unbabel/wmt22-comet-da`: Quality estimation model
- `masakhane/africomet-mtl`:  African languages model (see more [here](https://huggingface.co/masakhane))

## Memory Management

The script includes several memory optimization features:

- Automatic GPU memory cleanup
- Gradient accumulation support
- Error recovery from OOM situations
- Efficient batch processing

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
--batch_size 4

# Reduce max sequence length
--max_length 256
```

**Missing Dependencies:**
```bash
# Install all optional dependencies
pip install sacrebleu unbabel-comet
```

**Language Code Errors:**
- Ensure you're using correct NLLB language codes
- Check the official NLLB documentation for supported languages

**Data Format Issues:**
- Verify JSON is valid using online validators
- Ensure required columns exist in your data
- Check for missing or null values

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Mixed Precision**: Consider adding mixed precision training for larger models
4. **Data Preprocessing**: Clean and normalize your data before training

## Contributing

Feel free to submit issues and enhancement requests. When contributing:

1. Follow the existing code style
2. Add appropriate comments and documentation
3. Test your changes thoroughly
4. Update the README if needed

## Acknowledgment

This script is built on David Dale's tutorial on [How to fine-tune a NLLB-200 model for translating a new language](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865) and the accompanying [Colab Notebook](https://colab.research.google.com/drive/1bayEaw2fz_9Mhg9jFFZhrmDlQlBj1YZf?usp=sharing). It was extended to support additional features like fine-tuning on existing language pairs, and flexible data formats, among others.

## Contributing

Contributions are welcome, especially on adding support for more pre-trained models! Please feel free to submit a Pull Request.

## References

- [NLLB Paper](https://arxiv.org/abs/2207.04672)
- [NLLB Models on Hugging Face](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [COMET Evaluation](https://github.com/Unbabel/COMET)
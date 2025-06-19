# NLLB Fine-tuning Script

A comprehensive Python script for fine-tuning NLLB-200 (No Language Left Behind) models on custom translation datasets with support for new language pairs and vocabulary extension.

## Features

- 🌍 **Multi-language Support**: Fine-tune on any language pair supported by NLLB-200
- 📚 **New Language Addition**: Extend vocabulary with new language tokens
- 📊 **Flexible Data Formats**: Support for JSON, JSONL, CSV, and TSV files
- 🔄 **Bidirectional Training**: Automatic bidirectional translation training
- 📈 **Smart Data Splitting**: Automatic train/dev/test splits or use pre-split data
- 🎯 **Prediction Pipeline**: Built-in evaluation on test sets
- 🛠️ **Memory Optimization**: GPU memory management and cleanup
- 📝 **Comprehensive Logging**: Detailed training progress and model information

## Requirements

```bash
pip install torch transformers sacremoses scikit-learn pandas numpy tqdm
```

## Quick Start

### Basic Fine-tuning

```bash
python nllb_finetune.py \
  --data_path data/translations.json \
  --src_col source_text \
  --tgt_col target_text \
  --src_lang eng_Latn \
  --tgt_lang fra_Latn \
  --output_dir ./fine_tuned_model
```

### Fine-tuning with New Language

```bash
python nllb_finetune.py \
  --data_path data/translations.json \
  --src_col source_text \
  --tgt_col target_text \
  --src_lang eng_Latn \
  --tgt_lang tyv_Cyrl \
  --new_lang tyv_Cyrl \
  --similar_lang rus_Cyrl \
  --output_dir ./fine_tuned_model
```

## Data Format

### Option 1: Single File with Splits

**JSON format:**
```json
[
  {
    "source_text": "Hello world",
    "target_text": "Bonjour le monde",
    "split": "train"
  },
  {
    "source_text": "How are you?",
    "target_text": "Comment allez-vous?",
    "split": "dev"
  }
]
```

**JSONL format:**
```jsonl
{"source_text": "Hello world", "target_text": "Bonjour le monde", "split": "train"}
{"source_text": "How are you?", "target_text": "Comment allez-vous?", "split": "dev"}
```

### Option 2: Separate Files

Use separate files for train, dev, and test splits:

```bash
python nllb_finetune.py \
  --train_file data/train.json \
  --dev_file data/dev.json \
  --test_file data/test.json \
  --src_col source_text \
  --tgt_col target_text \
  --src_lang eng_Latn \
  --tgt_lang fra_Latn \
  --output_dir ./fine_tuned_model \
  --do_predict
```

## Command Line Arguments

### Data Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--data_path` | str | Path to single data file (JSON/JSONL/CSV/TSV) |
| `--train_file` | str | Path to training data file |
| `--dev_file` | str | Path to development data file |
| `--test_file` | str | Path to test data file |
| `--src_col` | str | **Required.** Source language column name |
| `--tgt_col` | str | **Required.** Target language column name |
| `--src_lang` | str | **Required.** Source language code (e.g., `eng_Latn`) |
| `--tgt_lang` | str | **Required.** Target language code (e.g., `fra_Latn`) |

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `facebook/nllb-200-distilled-600M` | Base NLLB model |
| `--new_lang` | str | None | New language code to add |
| `--similar_lang` | str | None | Similar language for initialization |
| `--output_dir` | str | **Required** | Output directory for fine-tuned model |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 16 | Training batch size |
| `--max_length` | int | 128 | Maximum sequence length |
| `--learning_rate` | float | 1e-4 | Learning rate |
| `--weight_decay` | float | 1e-3 | Weight decay |
| `--warmup_steps` | int | 1000 | Number of warmup steps |
| `--training_steps` | int | None | Total training steps |
| `--epochs` | int | 3 | Number of epochs (if training_steps is None) |

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--do_predict` | flag | False | Run prediction on test set |
| `--test_size` | float | 0.1 | Test set size for auto-split |
| `--dev_size` | float | 0.1 | Dev set size for auto-split |

### Other Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--log_interval` | int | 1000 | Steps between logging |
| `--save_interval` | int | 5000 | Steps between model saves |
| `--seed` | int | 42 | Random seed |
| `--test_translation` | str | None | Test translation after training |

## Language Codes

NLLB uses specific language codes. Common examples:

- English: `eng_Latn`
- Hausa: `hau_Latn`
- Zulu: `zul_Latn`
- Amharic: `amh_Ethi`
- Arabic: `ara_Arab`

For a complete list, see the [NLLB-200 documentation](https://github.com/facebookresearch/fairseq/tree/nllb).

## Examples

### 1. Fine-tune English to French

```bash
python finetune.py \
  --model_name facebook/nllb-200-distilled-600M \
  --data_path data/en_fr_translations.json \
  --src_col english \
  --tgt_col french \
  --src_lang eng_Latn \
  --tgt_lang fra_Latn \
  --output_dir ./models/en_fr_model \
  --batch_size 32 \
  --epochs 5
```

### 2. Add New Language (Tuvan)

```bash
python finetune.py \
  --model_name facebook/nllb-200-distilled-600M \
  --train_file data/train.jsonl \
  --dev_file data/dev.jsonl \
  --src_col russian \
  --tgt_col tuvan \
  --src_lang rus_Cyrl \
  --tgt_lang tyv_Cyrl \
  --new_lang tyv_Cyrl \
  --similar_lang rus_Cyrl \
  --output_dir ./models/ru_tyv_model \
  --do_predict \
  --test_file data/test.jsonl
```

### 3. Large-scale Training

```bash
python finetune.py \
  --model_name facebook/nllb-200-distilled-600M \
  --data_path data/large_dataset.csv \
  --src_col source \
  --tgt_col target \
  --src_lang eng_Latn \
  --tgt_lang deu_Latn \
  --output_dir ./models/en_de_large \
  --batch_size 8 \
  --training_steps 50000 \
  --learning_rate 5e-5 \
  --save_interval 2500 \
  --test_translation "Hello, how are you today?"
```

## Output Files

After training, the script saves:

- **Model files**: PyTorch model weights and configuration
- **Tokenizer files**: Updated tokenizer with any new language tokens  
- **training_info.json**: Training losses and configuration
- **test_predictions.json**: Predictions on test set (if `--do_predict` used)

## Memory Management

The script includes automatic memory management:

- GPU memory cleanup after errors
- Gradient accumulation for large batches
- Efficient data loading and preprocessing

For very large models or limited GPU memory, consider:
- Reducing `--batch_size`
- Using gradient checkpointing
- Using smaller NLLB variants (e.g., `nllb-200-distilled-600M`)

## Advanced Usage

### Custom Text Preprocessing

The script applies NLLB-standard preprocessing:
- Moses punctuation normalization
- Non-printing character removal
- Unicode NFKC normalization

### Bidirectional Training

Training automatically includes both translation directions (src→tgt and tgt→src) for better performance.

### Vocabulary Extension

When adding new languages:
1. New language token is added to vocabulary
2. Token embeddings are initialized from similar language (if provided)
3. Model is resized to accommodate new tokens

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--max_length`
2. **Language code not found**: Check NLLB-200 supported languages
3. **Data format errors**: Ensure JSON/JSONL is properly formatted
4. **Missing columns**: Verify `--src_col` and `--tgt_col` match your data

### Performance Tips

- Use GPU with sufficient memory (8GB+ recommended)
- Start with smaller models for testing
- Use appropriate batch sizes for your hardware
- Monitor training loss for convergence

## Acknowledgment

This script is built on David Dale's tutorial on [How to fine-tune a NLLB-200 model for translating a new language](https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865) and the accompanying [Colab Notebook](https://colab.research.google.com/drive/1bayEaw2fz_9Mhg9jFFZhrmDlQlBj1YZf?usp=sharing). It was extended to support additional features like fine-tuning on existing language pairs, and flexible data formats, among others.

## Contributing

Contributions are welcome, especially on adding support for more pre-trained models! Please feel free to submit a Pull Request.
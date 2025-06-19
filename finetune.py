#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLLB Fine-tuning and Evaluation Script
Fine-tune NLLB-200 models for new language pairs with optional vocabulary extension.
Supports JSON/JSONL data formats for train, test, and dev splits.
Can be used for training, evaluation, or inference only.
Enhanced with multiple evaluation metrics: BLEU, chrF, COMET, etc.
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import typing as tp
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sacremoses import MosesPunctNormalizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    get_constant_schedule_with_warmup
)
from transformers.optimization import Adafactor

# Evaluation metrics imports
try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("Warning: sacrebleu not available. Install with: pip install sacrebleu")

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: comet-ml not available. Install with: pip install unbabel-comet")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Handles calculation of various translation metrics"""
    
    def __init__(self, metrics=None, comet_model=None):
        self.metrics = metrics or ['exact_match']
        self.comet_model_path = comet_model
        self.comet_model = None
        
        # Initialize COMET model if requested
        if 'comet' in self.metrics and COMET_AVAILABLE and self.comet_model_path:
            try:
                logger.info(f"Loading COMET model: {self.comet_model_path}")
                self.comet_model = load_from_checkpoint(self.comet_model_path)
                logger.info("COMET model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load COMET model: {e}")
                self.metrics = [m for m in self.metrics if m != 'comet']
    
    def calculate_metrics(self, predictions, references, sources=None):
        """Calculate all requested metrics"""
        results = {}
        
        # Exact match accuracy
        if 'exact_match' in self.metrics:
            exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
            results['exact_match'] = exact_matches / len(predictions) if predictions else 0
            results['exact_matches_count'] = exact_matches
        
        # BLEU score
        if 'bleu' in self.metrics and SACREBLEU_AVAILABLE:
            try:
                bleu = sacrebleu.corpus_bleu(predictions, [references])
                results['bleu'] = bleu.score
                results['bleu_detailed'] = {
                    'score': bleu.score,
                    'counts': bleu.counts,
                    'totals': bleu.totals,
                    'precisions': bleu.precisions,
                    'bp': bleu.bp,
                    'sys_len': bleu.sys_len,
                    'ref_len': bleu.ref_len
                }
            except Exception as e:
                logger.warning(f"BLEU calculation failed: {e}")
                results['bleu'] = 0
        
        # chrF score
        if 'chrf' in self.metrics and SACREBLEU_AVAILABLE:
            try:
                chrf = sacrebleu.corpus_chrf(predictions, [references])
                results['chrf'] = chrf.score
                results['chrf_detailed'] = {
                    'score': chrf.score,
                    'nrefs': chrf.nrefs,
                    'char_order': chrf.char_order,
                    'word_order': chrf.word_order,
                    'beta': chrf.beta
                }
            except Exception as e:
                logger.warning(f"chrF calculation failed: {e}")
                results['chrf'] = 0
        
        # chrF++ score
        if 'chrfpp' in self.metrics and SACREBLEU_AVAILABLE:
            try:
                chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
                results['chrfpp'] = chrfpp.score
                results['chrfpp_detailed'] = {
                    'score': chrfpp.score,
                    'nrefs': chrfpp.nrefs,
                    'char_order': chrfpp.char_order,
                    'word_order': chrfpp.word_order,
                    'beta': chrfpp.beta
                }
            except Exception as e:
                logger.warning(f"chrF++ calculation failed: {e}")
                results['chrfpp'] = 0
        
        # TER score
        if 'ter' in self.metrics and SACREBLEU_AVAILABLE:
            try:
                ter = sacrebleu.corpus_ter(predictions, [references])
                results['ter'] = ter.score
                results['ter_detailed'] = {
                    'score': ter.score,
                    'num_edits': ter.num_edits,
                    'ref_len': ter.ref_len
                }
            except Exception as e:
                logger.warning(f"TER calculation failed: {e}")
                results['ter'] = 0
        
        # COMET score
        if 'comet' in self.metrics and self.comet_model and sources:
            try:
                # Prepare data for COMET
                comet_data = []
                for src, pred, ref in zip(sources, predictions, references):
                    comet_data.append({
                        "src": src,
                        "mt": pred,
                        "ref": ref
                    })
                
                # Calculate COMET scores
                comet_scores = self.comet_model.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
                results['comet'] = np.mean(comet_scores.scores)
                results['comet_scores'] = comet_scores.scores
                results['comet_system_score'] = comet_scores.system_score
                
            except Exception as e:
                logger.warning(f"COMET calculation failed: {e}")
                results['comet'] = 0
        
        # Length statistics
        if 'length_stats' in self.metrics:
            pred_lens = [len(p.split()) for p in predictions]
            ref_lens = [len(r.split()) for r in references]
            
            results['length_stats'] = {
                'pred_avg_length': np.mean(pred_lens),
                'ref_avg_length': np.mean(ref_lens),
                'length_ratio': np.mean(pred_lens) / np.mean(ref_lens) if np.mean(ref_lens) > 0 else 0,
                'pred_std_length': np.std(pred_lens),
                'ref_std_length': np.std(ref_lens)
            }
        
        return results


class NLLBTrainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.losses = []
        
        # Setup text preprocessing
        self.mpn = MosesPunctNormalizer(lang="en")
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = self._get_non_printing_char_replacer(" ")
        
        # Setup metrics calculator
        self.metrics_calculator = MetricsCalculator(
            metrics=args.metrics if hasattr(args, 'metrics') else ['exact_match'],
            comet_model=args.comet_model if hasattr(args, 'comet_model') else None
        )
        
        # Set random seeds for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    def _get_non_printing_char_replacer(self, replace_by: str = " ") -> tp.Callable[[str], str]:
        """Create function to replace non-printing characters"""
        non_printable_map = {
            ord(c): replace_by
            for c in (chr(i) for i in range(sys.maxunicode + 1))
            if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
        }

        def replace_non_printing_char(line) -> str:
            return line.translate(non_printable_map)

        return replace_non_printing_char
    
    def preprocess_text(self, text):
        """Preprocess text using NLLB normalization"""
        clean = self.mpn.normalize(text)
        clean = self.replace_nonprint(clean)
        clean = unicodedata.normalize("NFKC", clean)
        return clean
    
    def fix_tokenizer(self, new_lang=None):
        """Add a new language token to the tokenizer vocabulary"""
        if new_lang is None:
            return
            
        old_len = len(self.tokenizer) - int(new_lang in self.tokenizer.added_tokens_encoder)
        self.tokenizer.lang_code_to_id[new_lang] = old_len - 1
        self.tokenizer.id_to_lang_code[old_len - 1] = new_lang
        
        # Always move "mask" to the last position
        self.tokenizer.fairseq_tokens_to_ids["<mask>"] = (
            len(self.tokenizer.sp_model) + 
            len(self.tokenizer.lang_code_to_id) + 
            self.tokenizer.fairseq_offset
        )

        self.tokenizer.fairseq_tokens_to_ids.update(self.tokenizer.lang_code_to_id)
        self.tokenizer.fairseq_ids_to_tokens = {
            v: k for k, v in self.tokenizer.fairseq_tokens_to_ids.items()
        }
        
        if new_lang not in self.tokenizer._additional_special_tokens:
            self.tokenizer._additional_special_tokens.append(new_lang)
            
        # Clear the added token encoder
        self.tokenizer.added_tokens_encoder = {}
        self.tokenizer.added_tokens_decoder = {}
        
        logger.info(f"Added new language token: {new_lang}")
    
    def load_json_data(self, file_path):
        """Load data from JSON or JSONL file"""
        data = []
        
        if file_path.endswith('.jsonl'):
            # Load JSONL (one JSON object per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif file_path.endswith('.json'):
            # Load JSON (single JSON array or object)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    data = json_data
                else:
                    # If it's a single object, wrap it in a list
                    data = [json_data]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return pd.DataFrame(data)
    
    def load_data(self):
        """Load and split the data from JSON/JSONL files"""
        
        # Check if separate files are provided
        if hasattr(self.args, 'train_file') and self.args.train_file:
            logger.info("Loading data from separate train/dev/test files")
            
            # Load training data (only if training)
            df_train = None
            if self.args.do_train:
                df_train = self.load_json_data(self.args.train_file)
                logger.info(f"Loaded training data from {self.args.train_file}: {len(df_train)} samples")
            
            # Load dev data if provided
            df_dev = None
            if hasattr(self.args, 'dev_file') and self.args.dev_file:
                df_dev = self.load_json_data(self.args.dev_file)
                logger.info(f"Loaded dev data from {self.args.dev_file}: {len(df_dev)} samples")
            
            # Load test data if needed
            df_test = None
            if (self.args.do_predict or self.args.do_eval) and hasattr(self.args, 'test_file') and self.args.test_file:
                df_test = self.load_json_data(self.args.test_file)
                logger.info(f"Loaded test data from {self.args.test_file}: {len(df_test)} samples")
            elif (self.args.do_predict or self.args.do_eval) and not self.args.test_file:
                logger.warning("do_predict/do_eval is enabled but no test_file provided")
        
        else:
            # Single file with splits
            if not self.args.data_path:
                if self.args.do_train:
                    raise ValueError("Training requires either --data_path or --train_file")
                else:
                    # If not training and no data files, return empty DataFrames
                    return None, None, None
                    
            logger.info(f"Loading data from {self.args.data_path}")
            
            if self.args.data_path.endswith(('.json', '.jsonl')):
                df = self.load_json_data(self.args.data_path)
            elif self.args.data_path.endswith('.tsv'):
                df = pd.read_csv(self.args.data_path, sep='\t')
            elif self.args.data_path.endswith('.csv'):
                df = pd.read_csv(self.args.data_path)
            else:
                raise ValueError("Data file must be .json, .jsonl, .csv or .tsv format")
            
            # Check if split column exists
            if 'split' in df.columns:
                df_train = df[df.split == 'train'].copy() if self.args.do_train else None
                df_dev = df[df.split == 'dev'].copy() if 'dev' in df.split.values else None
                df_test = df[df.split == 'test'].copy() if ((self.args.do_predict or self.args.do_eval) and 'test' in df.split.values) else None
            else:
                # Create splits only if training
                if self.args.do_train:
                    if self.args.do_predict or self.args.do_eval:
                        # Include test set in split
                        df_train, df_temp = train_test_split(
                            df, test_size=self.args.test_size + self.args.dev_size, 
                            random_state=self.args.seed
                        )
                        if self.args.dev_size > 0:
                            df_dev, df_test = train_test_split(
                                df_temp, test_size=self.args.test_size / (self.args.test_size + self.args.dev_size),
                                random_state=self.args.seed
                            )
                        else:
                            df_dev = None
                            df_test = df_temp
                    else:
                        # Only create train/dev split
                        if self.args.dev_size > 0:
                            df_train, df_dev = train_test_split(
                                df, test_size=self.args.dev_size, 
                                random_state=self.args.seed
                            )
                        else:
                            df_train = df
                            df_dev = None
                        df_test = None
                else:
                    # Not training, use all data as test
                    df_train = None
                    df_dev = None
                    df_test = df if (self.args.do_predict or self.args.do_eval) else None
        
        # Drop rows with missing values
        if df_train is not None:
            df_train = df_train.dropna(subset=[self.args.src_col, self.args.tgt_col])
            logger.info(f"Final training samples: {len(df_train)}")
            
        if df_dev is not None:
            df_dev = df_dev.dropna(subset=[self.args.src_col, self.args.tgt_col])
            logger.info(f"Final development samples: {len(df_dev)}")
            
        if df_test is not None:
            df_test = df_test.dropna(subset=[self.args.src_col, self.args.tgt_col])
            logger.info(f"Final test samples: {len(df_test)}")
        
        return df_train, df_dev, df_test
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        # Determine model path - use existing model if available, otherwise base model
        model_path = self.args.model_name
        if self.args.model_path and os.path.exists(self.args.model_path):
            model_path = self.args.model_path
            logger.info(f"Loading existing model from: {model_path}")
        else:
            logger.info(f"Loading base model: {model_path}")
        
        self.tokenizer = NllbTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Add new language if specified (only for training)
        if self.args.new_lang and self.args.do_train:
            self.fix_tokenizer(self.args.new_lang)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Initialize new language token with similar language if provided
            if self.args.similar_lang:
                added_token_id = self.tokenizer.convert_tokens_to_ids(self.args.new_lang)
                similar_lang_id = self.tokenizer.convert_tokens_to_ids(self.args.similar_lang)
                
                # Move mask embedding to new position
                self.model.model.shared.weight.data[added_token_id + 1] = \
                    self.model.model.shared.weight.data[added_token_id]
                # Initialize new language with similar language embedding
                self.model.model.shared.weight.data[added_token_id] = \
                    self.model.model.shared.weight.data[similar_lang_id]
                
                logger.info(f"Initialized {self.args.new_lang} with {self.args.similar_lang} embedding")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            logger.info("Model moved to GPU")
    
    def setup_optimizer(self, training_steps):
        """Setup optimizer and scheduler"""
        self.optimizer = Adafactor(
            [p for p in self.model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=self.args.learning_rate,
            clip_threshold=1.0,
            weight_decay=self.args.weight_decay,
        )
        
        self.scheduler = get_constant_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.args.warmup_steps
        )
        
        logger.info(f"Setup optimizer with lr={self.args.learning_rate}")
    
    def get_batch_pairs(self, batch_size, data):
        """Generate a batch of translation pairs"""
        lang_pairs = [
            (self.args.src_col, self.args.src_lang),
            (self.args.tgt_col, self.args.tgt_lang)
        ]
        
        # Randomly choose direction
        (l1, long1), (l2, long2) = random.sample(lang_pairs, 2)
        
        xx, yy = [], []
        for _ in range(batch_size):
            item = data.iloc[random.randint(0, len(data) - 1)]
            xx.append(self.preprocess_text(str(item[l1])))
            yy.append(self.preprocess_text(str(item[l2])))
        
        return xx, yy, long1, long2
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def evaluate_dataset(self, df_eval, dataset_name="evaluation"):
        """Evaluate model on a dataset and compute metrics"""
        if df_eval is None:
            logger.warning(f"No {dataset_name} data available")
            return None
        
        logger.info(f"Running evaluation on {len(df_eval)} {dataset_name} samples")
        
        self.model.eval()
        predictions = []
        references = []
        sources = []
        
        with torch.no_grad():
            for idx, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=f"Evaluating {dataset_name}"):
                source_text = self.preprocess_text(str(row[self.args.src_col]))
                target_text = self.preprocess_text(str(row[self.args.tgt_col]))
                
                # Translate
                try:
                    translated = self.translate(
                        source_text, 
                        self.args.src_lang, 
                        self.args.tgt_lang
                    )[0]
                except Exception as e:
                    logger.warning(f"Translation failed for sample {idx}: {e}")
                    translated = ""
                
                predictions.append(translated)
                references.append(target_text)
                sources.append(source_text)
        
        # Calculate all requested metrics
        logger.info("Calculating metrics...")
        metrics_results = self.metrics_calculator.calculate_metrics(
            predictions=predictions,
            references=references,
            sources=sources
        )
        
        results = {
            'dataset': dataset_name,
            'total_samples': len(predictions),
            'predictions': predictions,
            'references': references,
            'sources': sources,
            'metrics': metrics_results
        }
        
        # Log results
        logger.info(f"{dataset_name.capitalize()} Results:")
        logger.info(f"  Total samples: {results['total_samples']}")
        
        for metric_name, metric_value in metrics_results.items():
            if not metric_name.endswith('_detailed') and not metric_name.endswith('_scores') and not metric_name.endswith('_count'):
                logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        return results
    
    def train(self, df_train):
        """Main training loop"""
        logger.info("Starting training...")
        
        training_steps = self.args.training_steps or len(df_train) // self.args.batch_size * self.args.epochs
        self.setup_optimizer(training_steps)
        
        self.model.train()
        self.cleanup_memory()
        
        progress_bar = trange(training_steps)
        
        for step in progress_bar:
            xx, yy, lang1, lang2 = self.get_batch_pairs(self.args.batch_size, df_train)
            
            try:
                # Tokenize source
                self.tokenizer.src_lang = lang1
                x = self.tokenizer(
                    xx, return_tensors='pt', padding=True, 
                    truncation=True, max_length=self.args.max_length
                ).to(self.model.device)
                
                # Tokenize target
                self.tokenizer.src_lang = lang2
                y = self.tokenizer(
                    yy, return_tensors='pt', padding=True, 
                    truncation=True, max_length=self.args.max_length
                ).to(self.model.device)
                
                # Mask padding tokens in labels
                y.input_ids[y.input_ids == self.tokenizer.pad_token_id] = -100

                # Forward pass
                loss = self.model(**x, labels=y.input_ids).loss
                loss.backward()
                self.losses.append(loss.item())

                # Optimization step
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            except RuntimeError as e:
                logger.warning(f"Runtime error at step {step}: {e}")
                self.optimizer.zero_grad(set_to_none=True)
                self.cleanup_memory()
                continue

            # Logging and saving
            if step % self.args.log_interval == 0 and step > 0:
                avg_loss = np.mean(self.losses[-self.args.log_interval:])
                progress_bar.set_description(f"Loss: {avg_loss:.4f}")
                logger.info(f"Step {step}, Average Loss: {avg_loss:.4f}")

            if step % self.args.save_interval == 0 and step > 0:
                self.save_model()
        
        logger.info("Training completed!")
        self.save_model()
    
    def save_model(self):
        """Save model and tokenizer"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        # Save training info
        info = {
            'losses': self.losses,
            'args': vars(self.args)
        }
        
        with open(os.path.join(self.args.output_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Model saved to {self.args.output_dir}")
    
    def save_evaluation_results(self, results, filename_prefix="evaluation"):
        """Save evaluation results to files"""
        if not results:
            return
            
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Save detailed results
        output_file = os.path.join(self.args.output_dir, f'{filename_prefix}_results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Evaluation results saved to {output_file}")
        
        # Save predictions in a more readable format
        predictions_file = os.path.join(self.args.output_dir, f'{filename_prefix}_predictions.txt')
        with open(predictions_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results for {results['dataset']}\n")
            f.write(f"Total samples: {results['total_samples']}\n")
            f.write("="*50 + "\n")
            
            # Write metric results
            f.write("METRICS:\n")
            for metric_name, metric_value in results['metrics'].items():
                if not metric_name.endswith('_detailed') and not metric_name.endswith('_scores') and not metric_name.endswith('_count'):
                    f.write(f"  {metric_name.upper()}: {metric_value:.4f}\n")
            f.write("="*50 + "\n\n")
            
            # Write sample-by-sample results
            for i, (src, ref, pred) in enumerate(zip(results['sources'], results['references'], results['predictions'])):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Source: {src}\n")
                f.write(f"Reference: {ref}\n")
                f.write(f"Prediction: {pred}\n")
                f.write(f"Match: {'✓' if pred.strip() == ref.strip() else '✗'}\n")
                f.write("-" * 30 + "\n")
        
        logger.info(f"Readable predictions saved to {predictions_file}")
        
        # Save metrics summary
        metrics_file = os.path.join(self.args.output_dir, f'{filename_prefix}_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results['metrics'], f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Metrics summary saved to {metrics_file}")
    
    def translate(self, text, src_lang, tgt_lang, **kwargs):
        """Translate text using the model"""
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        inputs = self.tokenizer(
            text, return_tensors='pt', padding=True, 
            truncation=True, max_length=self.args.max_length
        )
        
        self.model.eval()
        with torch.no_grad():
            result = self.model.generate(
                **inputs.to(self.model.device),
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                max_new_tokens=int(32 + 2.0 * inputs.input_ids.shape[1]),
                num_beams=kwargs.get('num_beams', 4),
                no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 3),
                **kwargs
            )
        
        return self.tokenizer.batch_decode(result, skip_special_tokens=True)
    
    def interactive_translate(self):
        """Interactive translation mode"""
        logger.info("Starting interactive translation mode. Type 'quit' to exit.")
        
        while True:
            try:
                text = input(f"\nEnter text to translate ({self.args.src_lang} -> {self.args.tgt_lang}): ")
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text.strip():
                    continue
                
                result = self.translate(text, self.args.src_lang, self.args.tgt_lang)
                print(f"Translation: {result[0]}")
                
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Translation error: {e}")
    
    def run(self):
        """Main execution method"""
        logger.info("Starting NLLB Fine-tuning and Evaluation")
        
        # Load data
        df_train, df_dev, df_test = self.load_data()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Training
        if self.args.do_train:
            if df_train is None:
                raise ValueError("No training data available")
            self.train(df_train)
        
        # Evaluation on dev set
        if self.args.do_eval and df_dev is not None:
            dev_results = self.evaluate_dataset(df_dev, "dev")
            if dev_results:
                self.save_evaluation_results(dev_results, "dev")
        
        # Evaluation/Prediction on test set
        if (self.args.do_eval or self.args.do_predict) and df_test is not None:
            test_results = self.evaluate_dataset(df_test, "test")
            if test_results:
                self.save_evaluation_results(test_results, "test")
        
        # Interactive mode
        if self.args.interactive:
            self.interactive_translate()


def main():
    parser = argparse.ArgumentParser(description="NLLB Fine-tuning and Evaluation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M",
                       help="Base model name or path")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to existing fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for model and results")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data file (JSON/JSONL/CSV/TSV)")
    parser.add_argument("--train_file", type=str, default=None,
                       help="Path to training data file")
    parser.add_argument("--dev_file", type=str, default=None,
                       help="Path to development data file")
    parser.add_argument("--test_file", type=str, default=None,
                       help="Path to test data file")
    parser.add_argument("--src_col", type=str, default="source",
                       help="Source text column name")
    parser.add_argument("--tgt_col", type=str, default="target",
                       help="Target text column name")
    parser.add_argument("--src_lang", type=str, required=True,
                       help="Source language code (e.g., 'eng_Latn')")
    parser.add_argument("--tgt_lang", type=str, required=True,
                       help="Target language code (e.g., 'fra_Latn')")
    
    # Language extension arguments
    parser.add_argument("--new_lang", type=str, default=None,
                       help="New language code to add to tokenizer")
    parser.add_argument("--similar_lang", type=str, default=None,
                       help="Similar language for initializing new language embedding")
    
    # Training arguments
    parser.add_argument("--do_train", action="store_true",
                       help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                       help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true",
                       help="Whether to run prediction")
    parser.add_argument("--interactive", action="store_true",
                       help="Run interactive translation mode")
    
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--training_steps", type=int, default=None,
                       help="Number of training steps (overrides epochs)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Data split arguments
    parser.add_argument("--test_size", type=float, default=0.1,
                       help="Test set size (fraction)")
    parser.add_argument("--dev_size", type=float, default=0.1,
                       help="Dev set size (fraction)")
    
    # Logging and saving arguments
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000,
                       help="Model saving interval")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Evaluation metrics arguments
    parser.add_argument("--metrics", type=str, nargs="+", 
                       default=["exact_match", "bleu", "chrf"],
                       choices=["exact_match", "bleu", "chrf", "chrfpp", "ter", "comet", "length_stats"],
                       help="Evaluation metrics to compute")
    parser.add_argument("--comet_model", type=str, default=None,
                       help="COMET model path or name (e.g., 'Unbabel/wmt22-comet-da')")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.do_train or args.do_eval or args.do_predict or args.interactive):
        parser.error("Must specify at least one of --do_train, --do_eval, --do_predict, or --interactive")
    
    if args.do_train and not (args.data_path or args.train_file):
        parser.error("Training requires either --data_path or --train_file")
    
    if (args.do_eval or args.do_predict) and not (args.data_path or args.test_file or args.dev_file):
        parser.error("Evaluation/prediction requires data files")
    
    if args.new_lang and not args.do_train:
        logger.warning("--new_lang specified but --do_train not set. New language will not be added.")
    
    # Download COMET model if specified
    if "comet" in args.metrics and args.comet_model and COMET_AVAILABLE:
        try:
            logger.info(f"Downloading COMET model: {args.comet_model}")
            args.comet_model = download_model(args.comet_model)
        except Exception as e:
            logger.error(f"Failed to download COMET model: {e}")
            args.metrics = [m for m in args.metrics if m != "comet"]
    
    # Initialize and run trainer
    trainer = NLLBTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
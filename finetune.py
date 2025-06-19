#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLLB Fine-tuning Script
Fine-tune NLLB-200 models for new language pairs with optional vocabulary extension.
Supports JSON/JSONL data formats for train, test, and dev splits.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        """Load and split the training data from JSON/JSONL files"""
        
        # Check if separate files are provided
        if hasattr(self.args, 'train_file') and self.args.train_file:
            logger.info("Loading data from separate train/dev/test files")
            
            # Load training data
            df_train = self.load_json_data(self.args.train_file)
            logger.info(f"Loaded training data from {self.args.train_file}: {len(df_train)} samples")
            
            # Load dev data if provided
            df_dev = None
            if hasattr(self.args, 'dev_file') and self.args.dev_file:
                df_dev = self.load_json_data(self.args.dev_file)
                logger.info(f"Loaded dev data from {self.args.dev_file}: {len(df_dev)} samples")
            
            # Load test data only if do_predict is enabled
            df_test = None
            if self.args.do_predict and hasattr(self.args, 'test_file') and self.args.test_file:
                df_test = self.load_json_data(self.args.test_file)
                logger.info(f"Loaded test data from {self.args.test_file}: {len(df_test)} samples")
            elif self.args.do_predict and not self.args.test_file:
                logger.warning("do_predict is enabled but no test_file provided")
        
        else:
            # Single file with splits
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
                df_train = df[df.split == 'train'].copy()
                df_dev = df[df.split == 'dev'].copy() if 'dev' in df.split.values else None
                df_test = df[df.split == 'test'].copy() if (self.args.do_predict and 'test' in df.split.values) else None
            else:
                # Create splits
                if self.args.do_predict:
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
        
        # Drop rows with missing values
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
        logger.info(f"Loading model: {self.args.model_name}")
        
        self.tokenizer = NllbTokenizer.from_pretrained(self.args.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name)
        
        # Add new language if specified
        if self.args.new_lang:
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
    
    def predict(self, df_test):
        """Run prediction on test set"""
        if df_test is None:
            logger.warning("No test data available for prediction")
            return
        
        logger.info(f"Running prediction on {len(df_test)} test samples")
        
        self.model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Predicting"):
                source_text = self.preprocess_text(str(row[self.args.src_col]))
                target_text = self.preprocess_text(str(row[self.args.tgt_col]))
                
                # Translate
                translated = self.translate(
                    source_text, 
                    self.args.src_lang, 
                    self.args.tgt_lang
                )[0]
                
                predictions.append(translated)
                references.append(target_text)
        
        # Save predictions
        results = {
            'predictions': predictions,
            'references': references,
            'sources': [self.preprocess_text(str(row[self.args.src_col])) for _, row in df_test.iterrows()]
        }
        
        output_file = os.path.join(self.args.output_dir, 'test_predictions.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Predictions saved to {output_file}")
        
        # Print a few examples
        logger.info("Sample predictions:")
        for i in range(min(5, len(predictions))):
            logger.info(f"Source: {results['sources'][i]}")
            logger.info(f"Reference: {references[i]}")
            logger.info(f"Prediction: {predictions[i]}")
            logger.info("---")
        """Translate text using the trained model"""
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


def main():
    parser = argparse.ArgumentParser(description='Fine-tune NLLB model for translation')
    
    # Data arguments - Option 1: Single file with splits
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data (JSON, JSONL, CSV or TSV)')
    
    # Data arguments - Option 2: Separate files
    parser.add_argument('--train_file', type=str, default=None,
                       help='Path to training data file (JSON or JSONL)')
    parser.add_argument('--dev_file', type=str, default=None,
                       help='Path to development data file (JSON or JSONL)')
    
    # Evaluation/Prediction arguments
    parser.add_argument('--do_predict', action='store_true',
                       help='Whether to run prediction on test set')
    parser.add_argument('--test_file', type=str, default=None,
                       help='Path to test data file (JSON or JSONL) - only used if do_predict is True')
    
    # Column/field names
    parser.add_argument('--src_col', type=str, required=True,
                       help='Name of source language field/column')
    parser.add_argument('--tgt_col', type=str, required=True,
                       help='Name of target language field/column')
    parser.add_argument('--src_lang', type=str, required=True,
                       help='Source language code (e.g., rus_Cyrl)')
    parser.add_argument('--tgt_lang', type=str, required=True,
                       help='Target language code (e.g., eng_Latn)')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='facebook/nllb-200-distilled-600M',
                       help='Base NLLB model to fine-tune')
    parser.add_argument('--new_lang', type=str, default=None,
                       help='New language code to add (e.g., tyv_Cyrl)')
    parser.add_argument('--similar_lang', type=str, default=None,
                       help='Similar language code for initialization')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the fine-tuned model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--training_steps', type=int, default=None,
                       help='Total training steps (if None, calculated from epochs)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (used if training_steps is None)')
    
    # Data splitting (if no split column exists and using single file)
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Test set size (if no split column)')
    parser.add_argument('--dev_size', type=float, default=0.1,
                       help='Dev set size (if no split column)')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=1000,
                       help='Steps between logging')
    parser.add_argument('--save_interval', type=int, default=5000,
                       help='Steps between model saves')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Testing
    parser.add_argument('--test_translation', type=str, default=None,
                       help='Test translation after training')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.data_path and not args.train_file:
        parser.error("Either --data_path or --train_file must be provided")
    
    if args.data_path and args.train_file:
        parser.error("Provide either --data_path OR --train_file, not both")
    
    if args.new_lang and not args.similar_lang:
        logger.warning("Adding new language without similar language initialization")
    
    # Initialize trainer
    trainer = NLLBTrainer(args)
    
    # Load data
    df_train, df_dev, df_test = trainer.load_data()
    
    # Setup model
    trainer.setup_model_and_tokenizer()
    
    # Train
    trainer.train(df_train)
    
    # Test translation if requested
    if args.test_translation:
        result = trainer.translate(
            args.test_translation, 
            args.src_lang, 
            args.tgt_lang
        )
        logger.info(f"Test translation: '{args.test_translation}' -> '{result[0]}'")


if __name__ == '__main__':
    main()
"""
Script to create a custom language model by fine-tuning a base model.
This allows you to train your own model based on your specific data.
"""

import os
import sys
import json
import argparse
import logging
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from config import settings
from utils import logger

# Configure logging for training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_dataset_from_json(file_path):
    """Load a dataset from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check format
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            # Extract text from each example
            texts = [item.get("text", "") for item in data["data"] if "text" in item]
            return Dataset.from_dict({"text": texts})
        else:
            logger.error(f"Invalid dataset format in {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def tokenize_data(dataset, tokenizer, max_length=512):
    """Tokenize dataset for training."""
    def tokenize_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels for causal language modeling (same as input_ids)
        result["labels"] = result["input_ids"].clone()
        return result
    
    # Apply tokenization to the dataset
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

def train_model(args):
    """Train a custom language model."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return
    
    # Load the dataset
    logger.info(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset_from_json(args.dataset)
    if dataset is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    try:
        # Load base model and tokenizer
        logger.info(f"Loading base model: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        # Check for existing checkpoint to resume training
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        checkpoint_exists = os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
        
        if checkpoint_exists and args.resume_training:
            # Find the latest checkpoint
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.base_model)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {device_name}")
            model = model.to('cuda')
        else:
            logger.info("GPU not available, using CPU")
        
        # Set pad_token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Process the dataset
        tokenized_dataset = tokenize_data(dataset, tokenizer)
        logger.info("Dataset tokenized successfully")
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "checkpoints"),
            overwrite_output_dir=not args.resume_training,  # Don't overwrite if resuming
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            save_strategy="epoch",
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=10,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            report_to="none",  # Disable tensorboard reporting
            # Disable multiprocessing to avoid Windows issues
            dataloader_num_workers=0,
            torch_compile=False
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model and tokenizer
        logger.info(f"Saving model to: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Create a README file to document the model
        with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"# Custom Voice Assistant Model\n\n")
            f.write(f"Base model: {args.base_model}\n")
            f.write(f"Training data: {args.dataset}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
            f.write(f"\nCreated on: {args.creation_date}\n")
        
        logger.info("Training complete!")
        
        # Print instructions for using the model
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo use your custom model:")
        print("1. Edit config/settings.py:")
        print("   - Set USE_CUSTOM_MODEL = True")
        print(f"   - Set CUSTOM_MODEL_PATH = '{args.output_dir}'")
        print(f"   - Set CUSTOM_TOKENIZER_PATH = '{args.output_dir}'")
        print("2. Run the voice assistant: python main.py")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Train a custom language model")
    parser.add_argument("--base_model", type=str, default="distilgpt2",
                       help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="data/training/sample_conversations.json",
                       help="Path to dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="models/custom_llm",
                       help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size (increase for GPU)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--resume_training", action="store_true",
                       help="Resume training from latest checkpoint")
    parser.add_argument("--creation_date", type=str, 
                       default=datetime.now().strftime("%Y-%m-%d"),
                       help="Creation date (automatic)")
    
    args = parser.parse_args()
    train_model(args)

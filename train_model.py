"""
Script to train a custom LLM model for the voice assistant.
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import settings
from utils import logger

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(settings.CUSTOM_MODEL_PATH, exist_ok=True)
    os.makedirs(settings.CUSTOM_TOKENIZER_PATH, exist_ok=True)
    os.makedirs(settings.TRAINING_DATA_PATH, exist_ok=True)
    logger.info(f"Created directories for model, tokenizer, and training data")

def train_model():
    """Train a custom language model based on the base model."""
    logger.info(f"Starting training process using {settings.MODEL_NAME} as base model")
    
    # Create necessary directories
    create_directories()
    
    try:
        # Check if training data exists
        data_file = f'{settings.TRAINING_DATA_PATH}/training_data.jsonl'
        if not os.path.exists(data_file):
            logger.critical(f"Training data not found at {data_file}")
            return False
        
        logger.info(f"Loading base model: {settings.MODEL_NAME}")
        
        # Load base model and tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        
        # If the tokenizer doesn't have a pad token, use the eos token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading dataset")
        # Load dataset
        dataset = load_dataset('json', data_files=data_file)
        logger.info(f"Loaded {len(dataset['train'])} training examples")
        
        # Define tokenize function
        def tokenize_function(examples):
            # Fix: flatten the input/output lists for batch processing
            inputs = [f"User: {inp}\nAssistant:" for inp in examples["input"]]
            targets = [out for out in examples["output"]]
            # Tokenize with padding and truncation for both inputs and labels
            model_inputs = tokenizer(
                inputs,
                padding="max_length",
                truncation=True,
                max_length=128
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    padding="max_length",
                    truncation=True,
                    max_length=128
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        logger.info("Tokenizing dataset")
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=settings.CUSTOM_MODEL_PATH,
            num_train_epochs=settings.TRAINING_EPOCHS,
            per_device_train_batch_size=settings.TRAINING_BATCH_SIZE,
            gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
            save_steps=settings.SAVE_STEPS,
            learning_rate=settings.LEARNING_RATE,
            logging_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='none',
            # Remove load_best_model_at_end and evaluation_strategy to avoid error
            # evaluation_strategy="no",
            # load_best_model_at_end=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        logger.info("Starting training process")
        # Train model
        trainer.train()
        
        logger.info("Training complete, saving model and tokenizer")
        # Save model and tokenizer
        trainer.save_model(settings.CUSTOM_MODEL_PATH)
        tokenizer.save_pretrained(settings.CUSTOM_TOKENIZER_PATH)
        
        logger.info(f"Model saved to {settings.CUSTOM_MODEL_PATH}")
        logger.info(f"Tokenizer saved to {settings.CUSTOM_TOKENIZER_PATH}")
        
        return True
        
    except Exception as e:
        logger.critical(f"Error during training: {e}")
        return False

def verify_model():
    """Test the trained model with a simple prompt."""
    try:
        logger.info("Verifying trained model")
        
        tokenizer = AutoTokenizer.from_pretrained(settings.CUSTOM_TOKENIZER_PATH)
        model = AutoModelForCausalLM.from_pretrained(settings.CUSTOM_MODEL_PATH)
        
        # Test prompt
        test_prompt = "User: Hello, what can you do?\nAssistant:"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        logger.info("Generating test response")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=100,
                do_sample=True,
                temperature=0.7
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test response: {response}")
        
        # Check if response contains more than just the prompt
        if len(response) > len(test_prompt):
            logger.info("Model verification successful!")
            return True
        else:
            logger.warning("Model response looks incomplete")
            return False
            
    except Exception as e:
        logger.critical(f"Error verifying model: {e}")
        return False

def update_config_for_custom_model():
    """Update settings to use the custom model."""
    try:
        # Read existing settings file
        with open(os.path.join(current_dir, 'config', 'settings.py'), 'r') as file:
            content = file.read()
        
        # Update USE_CUSTOM_MODEL setting to True
        if 'USE_CUSTOM_MODEL = False' in content:
            content = content.replace('USE_CUSTOM_MODEL = False', 'USE_CUSTOM_MODEL = True')
            
            # Write updated content back
            with open(os.path.join(current_dir, 'config', 'settings.py'), 'w') as file:
                file.write(content)
                
            logger.info("Updated settings to use custom model")
            return True
        else:
            logger.warning("Could not locate USE_CUSTOM_MODEL setting or it's already set to True")
            return False
            
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return False

if __name__ == "__main__":
    logger.info("Custom LLM training process started")
    
    success = train_model()
    
    if success:
        logger.info("Training completed successfully")
        
        # Verify the model
        if verify_model():
            # Update configuration to use custom model
            update_config_for_custom_model()
            logger.info("Model is ready to use with the voice assistant")
        else:
            logger.warning("Model verification failed. Check the model and try again")
    else:
        logger.critical("Training process failed")

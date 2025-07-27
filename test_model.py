"""
Script to test the custom LLM model.
"""

import os
import sys
import torch

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import settings
from utils import logger

def test_model():
    """Test the model with various prompts."""
    try:
        # Determine which model to test
        if settings.USE_CUSTOM_MODEL and os.path.exists(settings.CUSTOM_MODEL_PATH):
            model_path = settings.CUSTOM_MODEL_PATH
            tokenizer_path = settings.CUSTOM_TOKENIZER_PATH
            logger.info(f"Testing custom model from {model_path}")
        else:
            model_path = settings.MODEL_NAME
            tokenizer_path = settings.MODEL_NAME
            logger.info(f"Testing default model: {model_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Use CPU for testing to avoid CUDA memory issues
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Test prompts
        test_prompts = [
            "Hello, how are you?",
            "What's the weather today?",
            "Tell me a joke",
            "Who are you?",
            "What can you do?"
        ]
        
        logger.info("Starting model testing with sample prompts:")
        
        for prompt in test_prompts:
            logger.info(f"Testing prompt: '{prompt}'")
            
            # Format input for the model
            input_text = f"User: {prompt}\nAssistant:"
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=100,
                    temperature=settings.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:", 1)[1].strip()
            else:
                response = full_response[len(input_text):].strip()
                if not response:
                    response = full_response.strip()
            
            logger.info(f"Response: '{response}'")
            print(f"\nQ: {prompt}")
            print(f"A: {response}\n")
            print("-" * 40)
            
        return True
            
    except Exception as e:
        logger.critical(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    logger.info("Model testing started")
    test_model()
    logger.info("Model testing completed")

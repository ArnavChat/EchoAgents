"""
Module to generate responses using a Hugging Face model.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
from utils import logger
from config import settings

# Global variables to store model and tokenizer
model = None
tokenizer = None

def is_accelerate_available():
    """Check if the accelerate package is available."""
    return False  # Simplify to avoid dependency on accelerate

def load_model():
    """
    Load the Hugging Face model and tokenizer.
    Returns:
        tuple: (model, tokenizer) tuple or (None, None) if loading failed
    """
    global model, tokenizer
    
    model_path = settings.CUSTOM_MODEL_PATH if settings.USE_CUSTOM_MODEL else settings.MODEL_NAME
    tokenizer_path = settings.CUSTOM_TOKENIZER_PATH if settings.USE_CUSTOM_MODEL else settings.MODEL_NAME
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set pad_token to eos_token to fix padding issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for tokenizer")
        
        # Simplified model loading for CPU
        model_kwargs = {}
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        logger.info("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        model = None
        tokenizer = None
        return None, None

def generate_response(prompt):
    """
    Generate a response using the loaded Hugging Face model.
    
    Args:
        prompt (str): The user's query
    
    Returns:
        str: The model's response
    """
    global model, tokenizer
    
    # Load the model if not loaded
    if model is None or tokenizer is None:
        load_model()
    
    try:
        # Check if we have a custom prompt template
        prompt_template_path = os.path.join(settings.CUSTOM_MODEL_PATH, "prompt_template.txt")
        custom_prompt = ""
        if os.path.exists(prompt_template_path) and settings.USE_CUSTOM_MODEL:
            try:
                with open(prompt_template_path, 'r', encoding='utf-8') as f:
                    custom_prompt = f.read().strip() + "\n\n"
                logger.info("Using custom prompt template")
            except:
                custom_prompt = ""
        
        # Format input for the model
        input_text = custom_prompt + f"User: {prompt}\nAssistant:"
        
        # Tokenize the input without padding to avoid issues
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move inputs to the appropriate device if model is on a specific device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_length=settings.MAX_LENGTH + len(inputs["input_ids"][0]),
                temperature=settings.TEMPERATURE,
                top_p=getattr(settings, "TOP_P", 0.92),
                top_k=getattr(settings, "TOP_K", 50),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Prevent repetition of 3-grams
            )
        
        # Decode and format response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response - improved parsing
        if "Assistant:" in full_response:
            # Find the last occurrence of "Assistant:"
            last_idx = full_response.rfind("Assistant:")
            response = full_response[last_idx:].split("Assistant:", 1)[1].strip()
        else:
            # Get the response after the input prompt
            response = full_response[len(input_text):].strip()
            if not response:  # Fallback if parsing fails
                response = full_response.strip()
        
        logger.info(f"Response generated: {response[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process that request."

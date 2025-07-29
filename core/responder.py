"""
Module to generate responses using a Hugging Face model.
"""

<<<<<<< HEAD
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
=======
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PhiForCausalLM
import importlib.util
import os
import json
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed
from utils import logger
from config import settings

# Global variables to store model and tokenizer
model = None
tokenizer = None

def is_accelerate_available():
    """Check if the accelerate package is available."""
<<<<<<< HEAD
    return False  # Simplify to avoid dependency on accelerate
=======
    return importlib.util.find_spec("accelerate") is not None

def ensure_model_type_in_config(config_path, original_model_name):
    """Ensure the config.json contains a proper model_type."""
    if not os.path.exists(config_path):
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check if model_type is missing
        if 'model_type' not in config:
            # Get model_type from original model
            original_config_path = os.path.join(current_dir, 'original_config.json')
            if os.path.exists(original_config_path):
                with open(original_config_path, 'r') as f:
                    original_config = json.load(f)
                model_type = original_config.get('model_type', 'phi')
            else:
                # Default to 'phi' for phi-2 models
                model_type = 'phi'
                
            # Add model_type to config
            config['model_type'] = model_type
            
            # Write updated config back
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Added missing model_type '{model_type}' to config.json")
            return True
    except Exception as e:
        logger.error(f"Error updating config.json: {e}")
        
    return False
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed

def load_model():
    """
    Load the Hugging Face model and tokenizer.
<<<<<<< HEAD
    Returns:
        tuple: (model, tokenizer) tuple or (None, None) if loading failed
    """
    global model, tokenizer
    
    model_path = settings.CUSTOM_MODEL_PATH if settings.USE_CUSTOM_MODEL else settings.MODEL_NAME
    tokenizer_path = settings.CUSTOM_TOKENIZER_PATH if settings.USE_CUSTOM_MODEL else settings.MODEL_NAME
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
=======
    """
    global model, tokenizer
    
    if settings.USE_CUSTOM_MODEL:
        logger.info(f"Loading custom model from: {settings.CUSTOM_MODEL_PATH}")
        try:
            # First check if the model and tokenizer directories exist
            if not os.path.exists(settings.CUSTOM_MODEL_PATH):
                raise ValueError(f"Custom model directory not found: {settings.CUSTOM_MODEL_PATH}")
            if not os.path.exists(settings.CUSTOM_TOKENIZER_PATH):
                raise ValueError(f"Custom tokenizer directory not found: {settings.CUSTOM_TOKENIZER_PATH}")
            
            # Load the tokenizer
            logger.info(f"Loading custom tokenizer from: {settings.CUSTOM_TOKENIZER_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(
                settings.CUSTOM_TOKENIZER_PATH,
                trust_remote_code=True
            )
            
            # Set pad_token to eos_token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token for tokenizer")
            
            # Load the model with additional trust_remote_code=True option
            logger.info(f"Loading custom model with trust_remote_code=True")
            model_kwargs = {"torch_dtype": torch.float16}
            
            if is_accelerate_available():
                model_kwargs["device_map"] = "auto"
                logger.info("Using accelerate for model loading with device_map")
                
            model = AutoModelForCausalLM.from_pretrained(
                settings.CUSTOM_MODEL_PATH,
                trust_remote_code=True,
                **model_kwargs
            )
            
            logger.info("Custom model loaded successfully")
            
        except Exception as e:
            logger.critical(f"Failed to load custom model: {e}")
            logger.warning("Falling back to default model")
            load_default_model()
    else:
        load_default_model()

def load_default_model():
    """Load the default Hugging Face model."""
    global model, tokenizer
    
    logger.info(f"Loading model: {settings.MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed
        
        # Set pad_token to eos_token to fix padding issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for tokenizer")
        
<<<<<<< HEAD
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
=======
        # Configure model loading based on available packages
        model_kwargs = {"torch_dtype": torch.float16}
        
        if is_accelerate_available():
            model_kwargs["device_map"] = "auto"
            logger.info("Using accelerate for model loading with device_map")
        else:
            logger.warning("Package 'accelerate' not found; using CPU only. Install with 'pip install accelerate'")
            
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            **model_kwargs
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed

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
<<<<<<< HEAD
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
=======
        # Format input for the model
        input_text = f"User: {prompt}\nAssistant:"
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed
        
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
<<<<<<< HEAD
                top_p=getattr(settings, "TOP_P", 0.92),
                top_k=getattr(settings, "TOP_K", 50),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Prevent repetition of 3-grams
=======
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed
            )
        
        # Decode and format response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response - improved parsing
        if "Assistant:" in full_response:
<<<<<<< HEAD
            # Find the last occurrence of "Assistant:"
            last_idx = full_response.rfind("Assistant:")
            response = full_response[last_idx:].split("Assistant:", 1)[1].strip()
=======
            response = full_response.split("Assistant:", 1)[1].strip()
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed
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
<<<<<<< HEAD
=======
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process that request."
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed

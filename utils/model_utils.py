"""
Utility functions for model management.
"""

import os
from config import settings
from utils import logger

def verify_model_directories():
    """
    Verify that the custom model directories exist and have necessary files.
    
    Returns:
        bool: True if directories and necessary files exist, False otherwise.
    """
    # Check custom model path
    if not os.path.exists(settings.CUSTOM_MODEL_PATH):
        logger.warning(f"Custom model path not found: {settings.CUSTOM_MODEL_PATH}")
        return False
        
    # Check custom tokenizer path
    if not os.path.exists(settings.CUSTOM_TOKENIZER_PATH):
        logger.warning(f"Custom tokenizer path not found: {settings.CUSTOM_TOKENIZER_PATH}")
        return False
    
    # For PyTorch models, check for model files
    if settings.CUSTOM_MODEL_TYPE == "pytorch":
        model_files = ["config.json"]
        # Check for either pytorch_model.bin or model.safetensors - newer models use safetensors
        if not (os.path.exists(os.path.join(settings.CUSTOM_MODEL_PATH, "pytorch_model.bin")) or 
                os.path.exists(os.path.join(settings.CUSTOM_MODEL_PATH, "model.safetensors"))):
            logger.warning(f"Required model file not found: pytorch_model.bin or model.safetensors")
            return False
            
        for file in model_files:
            if not os.path.exists(os.path.join(settings.CUSTOM_MODEL_PATH, file)):
                logger.warning(f"Required model file not found: {file}")
                return False
                
        # Check tokenizer files - different models use different file names
        tokenizer_files = ["tokenizer_config.json", "special_tokens_map.json"]  # More flexible check
        for file in tokenizer_files:
            if not os.path.exists(os.path.join(settings.CUSTOM_TOKENIZER_PATH, file)):
                logger.warning(f"Required tokenizer file not found: {file}")
                return False
    
    logger.info("Custom model directories and files verified")
    return True

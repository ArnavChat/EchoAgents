"""
Script to fix the config.json of a custom model to ensure it has the proper model_type.
"""

import os
import json
import sys
from utils import logger

def fix_config_files():
    """Fix the config.json files to include model_type."""
    model_dir = "models\custom_llm"
    tokenizer_dir = "models\custom_tokenizer"
    
    # Fix model config
    model_config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(model_config_path):
        logger.info(f"Fixing model config at: {model_config_path}")
        try:
            with open(model_config_path, 'r') as f:
                config = json.load(f)
            
            # Add model_type if missing
            if 'model_type' not in config:
                config['model_type'] = 'phi'
                logger.info("Added 'model_type': 'phi' to model config")
                
                with open(model_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error fixing model config: {e}")
    else:
        logger.error(f"Model config not found at: {model_config_path}")
    
    # Fix tokenizer config
    tokenizer_config_path = os.path.join(tokenizer_dir, "config.json")
    if os.path.exists(tokenizer_config_path):
        logger.info(f"Fixing tokenizer config at: {tokenizer_config_path}")
        try:
            with open(tokenizer_config_path, 'r') as f:
                config = json.load(f)
            
            # Add model_type if missing
            if 'model_type' not in config:
                config['model_type'] = 'phi'
                logger.info("Added 'model_type': 'phi' to tokenizer config")
                
                with open(tokenizer_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error fixing tokenizer config: {e}")
    else:
        logger.error(f"Tokenizer config not found at: {tokenizer_config_path}")

if __name__ == "__main__":
    # Ensure we can import the logger
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    fix_config_files()
    print("Config files fixed. Try running the voice assistant now.")

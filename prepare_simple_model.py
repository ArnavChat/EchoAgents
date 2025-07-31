"""
Simple script to demonstrate using a small language model with your voice assistant.
This avoids the complexity of full training by using a pre-trained model.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import logger
from config import settings

def prepare_custom_model():
    """
    Download and prepare a small language model suitable for the voice assistant.
    This simulates having a custom model without the complexity of training.
    """
    # Create output directory
    os.makedirs(settings.CUSTOM_MODEL_PATH, exist_ok=True)
    
    # Model to use (small and fast)
    model_name = "distilgpt2"  # Only 82M parameters vs 2.7B for Phi-2
    
    try:
        logger.info(f"Downloading model: {model_name}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Save to custom model path
        logger.info(f"Saving model to {settings.CUSTOM_MODEL_PATH}")
        model.save_pretrained(settings.CUSTOM_MODEL_PATH)
        tokenizer.save_pretrained(settings.CUSTOM_MODEL_PATH)
        
        # Test the model with a simple prompt
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        test_prompt = "Hello, how are you?"
        response = generator(test_prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
        
        logger.info(f"Test generation successful: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing model: {e}")
        return False

def customize_with_examples(examples_file):
    """
    Customize the prompt template using examples from the sample conversations.
    This creates a prompt.txt file that can be used to improve responses.
    """
    try:
        # Read examples
        with open(examples_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = data.get("data", [])
        if not examples:
            logger.error(f"No examples found in {examples_file}")
            return False
        
        # Create a prompt template with examples
        prompt_template = "# Voice Assistant Conversation Examples\n\n"
        
        for example in examples:
            prompt_template += example.get("text", "") + "\n\n"
        
        # Save the prompt template
        prompt_file = os.path.join(settings.CUSTOM_MODEL_PATH, "prompt_template.txt")
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt_template)
        
        logger.info(f"Created prompt template with {len(examples)} examples at {prompt_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error customizing with examples: {e}")
        return False

def main():
    """Main function."""
    print("Preparing a simple custom model for your voice assistant...")
    
    # Prepare the model
    if prepare_custom_model():
        # Customize with examples
        examples_file = "data/training/sample_conversations.json"
        if customize_with_examples(examples_file):
            # Instructions
            print("\n" + "-" * 60)
            print("SUCCESS! Model prepared and customized with your examples.")
            print("\nTo use your custom model:")
            print("1. Edit config/settings.py:")
            print("   - Set USE_CUSTOM_MODEL = True")
            print(f"   - Ensure CUSTOM_MODEL_PATH = '{settings.CUSTOM_MODEL_PATH}'")
            print(f"   - Ensure CUSTOM_TOKENIZER_PATH = '{settings.CUSTOM_MODEL_PATH}'")
            print("2. Run the voice assistant: python main.py")
            print("-" * 60)
        else:
            print("Failed to customize model with examples.")
    else:
        print("Failed to prepare custom model.")

if __name__ == "__main__":
    main()

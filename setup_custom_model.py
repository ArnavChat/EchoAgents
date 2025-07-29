"""
Wrapper script to train, test, and integrate a custom LLM model.
"""

import os
os.environ["HF_HOME"] = "D:/hf_cache"
import sys
import subprocess
import time
from utils import logger

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        "torch", 
        "transformers", 
        "datasets", 
        "accelerate"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        for package in missing_packages:
            subprocess.call([sys.executable, "-m", "pip", "install", package])
        
        logger.info("Required packages installed")
    else:
        logger.info("All required packages are already installed")

def setup_directories():
    """Set up necessary directories for training and models."""
    directories = [
        "data/training",
        "models/custom_llm",
        "models/custom_tokenizer"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("Directory structure set up")

def run_training():
    """Run the model training process."""
    logger.info("Starting model training...")
    result = subprocess.call([sys.executable, "train_model.py"])
    
    if result == 0:
        logger.info("Model training completed successfully")
        return True
    else:
        logger.error("Model training failed")
        return False

def test_model():
    """Test the trained model."""
    logger.info("Testing trained model...")
    result = subprocess.call([sys.executable, "test_model.py"])
    
    if result == 0:
        logger.info("Model testing completed successfully")
        return True
    else:
        logger.error("Model testing failed")
        return False

def check_voice_assistant():
    """Check if the voice assistant works with the custom model."""
    logger.info("Checking voice assistant with custom model...")
    
    try:
        # Run voice assistant in a separate process for a short time
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for 10 seconds to see if it crashes
        time.sleep(10)
        
        # If process is still running, assume it's working
        if process.poll() is None:
            logger.info("Voice assistant is running properly with the custom model")
            process.terminate()
            return True
        else:
            # Get error output
            stdout, stderr = process.communicate()
            logger.error(f"Voice assistant failed to run: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking voice assistant: {e}")
        return False

def main():
    """Main function to run the entire setup process."""
    logger.info("Starting custom model setup process")
    
    # Step 1: Check requirements
    check_requirements()
    
    # Step 2: Set up directories
    setup_directories()
    
    # Step 3: Run training
    if run_training():
        # Step 4: Test the model
        if test_model():
            # Step 5: Check if voice assistant works
            if check_voice_assistant():
                logger.info("Custom model setup completed successfully!")
                return True
    
    logger.error("Custom model setup failed")
    return False

if __name__ == "__main__":
    main()

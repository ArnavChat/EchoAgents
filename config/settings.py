"""
Configuration settings for the voice assistant.
"""

# Model settings
MODEL_NAME = "distilgpt2"  # Fallback model if custom model fails
MAX_LENGTH = 75  # Reduced to avoid overly long responses
TEMPERATURE = 0.6  # Reduced for more focused answers
TOP_P = 0.92  # Nucleus sampling parameter
TOP_K = 50  # Limits vocabulary to top K tokens

# Voice settings
VOICE_RATE = 150  # Words per minute
VOICE_VOLUME = 1.0  # Volume level (0.0 to 1.0)
VOICE_GENDER = 'female'  # 'male' or 'female'

# Listener settings
LANGUAGE = "en-US"
PAUSE_THRESHOLD = 0.8  # Seconds of non-speaking audio before a phrase is considered complete

# App settings
COOLDOWN_TIME = 1.0  # Seconds to pause between speech cycles
DEBUG_MODE = True  # Enable additional debug output

# Custom LLM model settings
USE_CUSTOM_MODEL = True  # Set to True after running train_model.py
CUSTOM_MODEL_TYPE = "pytorch"  # Options: "pytorch", "tensorflow", "ggml"
CUSTOM_MODEL_PATH = "models/custom_llm_improved"  # Directory for your custom model files
CUSTOM_TOKENIZER_PATH = "models/custom_llm_improved"  # Path for custom tokenizer (same as model by default)

# Training settings for custom models
TRAINING_DATA_PATH = "data/training"  # Path to training data
TRAINING_EPOCHS = 3  # Number of training epochs
TRAINING_BATCH_SIZE = 2  # Training batch size 
LEARNING_RATE = 5e-5  # Learning rate for training
GRADIENT_ACCUMULATION_STEPS = 8  # For handling larger batch sizes on smaller GPUs
SAVE_STEPS = 500  # Save model every X steps during training

# Use a smaller model for better performance
BASE_MODEL_NAME = "distilgpt2"

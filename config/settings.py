"""
Configuration settings for the voice assistant.
"""

# Hugging Face model settings
MODEL_NAME = "microsoft/phi-2"
MAX_LENGTH = 250
TEMPERATURE = 0.7

# Voice settings
VOICE_RATE = 250  # Words per minute
VOICE_VOLUME = 1.0  # Volume level (0.0 to 1.0)
VOICE_GENDER = 'female'  # 'male' or 'female'

# Listener settings
LANGUAGE = "en-US"
PAUSE_THRESHOLD = 2  # Seconds of non-speaking audio before a phrase is considered complete

# App settings
COOLDOWN_TIME = 1.0  # Seconds to pause between speech cycles
DEBUG_MODE = True  # Enable additional debug output

# Custom LLM model settings
USE_CUSTOM_MODEL = True  # Changed from False to True to use your own model
CUSTOM_MODEL_TYPE = "pytorch"  # Options: "pytorch", "tensorflow", "ggml"
CUSTOM_MODEL_PATH = "models/custom_llm"  # Directory for your custom model files
CUSTOM_TOKENIZER_PATH = "models/custom_tokenizer"  # Path for custom tokenizer

# Training settings for custom models
TRAINING_DATA_PATH = "data/training"  # Path to training data
TRAINING_EPOCHS = 3  # Number of training epochs
TRAINING_BATCH_SIZE = 2  # Training batch size 
LEARNING_RATE = 5e-5  # Learning rate for training
GRADIENT_ACCUMULATION_STEPS = 8  # For handling larger batch sizes on smaller GPUs
SAVE_STEPS = 500  # Save model every X steps during training

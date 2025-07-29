"""
Configuration settings for the voice assistant.
"""

<<<<<<< HEAD
# Model settings
MODEL_NAME = "distilgpt2"  # Fallback model if custom model fails
MAX_LENGTH = 75  # Reduced to avoid overly long responses
TEMPERATURE = 0.6  # Reduced for more focused answers
TOP_P = 0.92  # Nucleus sampling parameter
TOP_K = 50  # Limits vocabulary to top K tokens

# Voice settings
VOICE_RATE = 150  # Words per minute
=======
# Hugging Face model settings
MODEL_NAME = "microsoft/phi-2"
MAX_LENGTH = 250
TEMPERATURE = 0.7

# Voice settings
VOICE_RATE = 250  # Words per minute
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed
VOICE_VOLUME = 1.0  # Volume level (0.0 to 1.0)
VOICE_GENDER = 'female'  # 'male' or 'female'

# Listener settings
LANGUAGE = "en-US"
<<<<<<< HEAD
PAUSE_THRESHOLD = 0.8  # Seconds of non-speaking audio before a phrase is considered complete
=======
PAUSE_THRESHOLD = 2  # Seconds of non-speaking audio before a phrase is considered complete
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed

# App settings
COOLDOWN_TIME = 1.0  # Seconds to pause between speech cycles
DEBUG_MODE = True  # Enable additional debug output

# Custom LLM model settings
<<<<<<< HEAD
USE_CUSTOM_MODEL = True  # Set to True after running train_model.py
CUSTOM_MODEL_TYPE = "pytorch"  # Options: "pytorch", "tensorflow", "ggml"
CUSTOM_MODEL_PATH = "models/custom_llm_improved"  # Directory for your custom model files
CUSTOM_TOKENIZER_PATH = "models/custom_llm_improved"  # Path for custom tokenizer (same as model by default)
=======
USE_CUSTOM_MODEL = True  # Changed from False to True to use your own model
CUSTOM_MODEL_TYPE = "pytorch"  # Options: "pytorch", "tensorflow", "ggml"
CUSTOM_MODEL_PATH = "models/custom_llm"  # Directory for your custom model files
CUSTOM_TOKENIZER_PATH = "models/custom_tokenizer"  # Path for custom tokenizer
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed

# Training settings for custom models
TRAINING_DATA_PATH = "data/training"  # Path to training data
TRAINING_EPOCHS = 3  # Number of training epochs
TRAINING_BATCH_SIZE = 2  # Training batch size 
LEARNING_RATE = 5e-5  # Learning rate for training
GRADIENT_ACCUMULATION_STEPS = 8  # For handling larger batch sizes on smaller GPUs
SAVE_STEPS = 500  # Save model every X steps during training
<<<<<<< HEAD

# Use a smaller model for better performance
BASE_MODEL_NAME = "distilgpt2"
=======
>>>>>>> 85bfd11344013a081217da2e24af47f81a98b9ed

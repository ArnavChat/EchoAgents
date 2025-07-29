"""
Main application for the desktop voi            # Force use fallback model
                config.settings.USE_CUSTOM_MODEL = False
                model_result, tokenizer_result = load_model()
                
                if model_result is None or tokenizer_result is None:
                    logger.critical("Failed to load fallback model. Exiting.")
                    return
        except Exception as e:
            logger.critical(f"Model loading error: {e}. Switching to fallback model...")
            # Force use fallback model
            config.settings.USE_CUSTOM_MODEL = False
            model_result, tokenizer_result = load_model().
"""

import time
import sys
import os
<<<<<<< HEAD
os.environ["HF_HOME"] = "D:/hf_cache"
=======
>>>>>>> 6b06759f49d127014a72d8bf03e13e30bc3aa987

# Add the project root to the path to make imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.listener import listen
from core.responder import load_model, generate_response
from core.speaker import speak
from utils import logger
from config.settings import COOLDOWN_TIME
import config.settings

# Import the model verification function
from utils.model_utils import verify_model_directories

def main():
    """Main function to run the voice assistant."""
    logger.info("Starting voice assistant...")
    
    try:
        # Verify model directories if using custom model
        if config.settings.USE_CUSTOM_MODEL:
            if not verify_model_directories():
                logger.warning("Custom model verification failed. Setting USE_CUSTOM_MODEL=False")
                config.settings.USE_CUSTOM_MODEL = False
        
        # Load AI model at startup
        logger.info("Loading AI model...")
        try:
            model_result, tokenizer_result = load_model()
            
            if model_result is None or tokenizer_result is None:
                logger.critical("Failed to load any model. Switching to default model...")
                # Force use default model
                config.settings.USE_CUSTOM_MODEL = False
                model_result, tokenizer_result = load_model()
                
                if model_result is None or tokenizer_result is None:
                    logger.critical("Failed to load default model. Exiting.")
                    return
        except Exception as e:
            logger.critical(f"Model loading error: {e}. Switching to default model...")
            # Force use default model on any error
            config.settings.USE_CUSTOM_MODEL = False
            model_result, tokenizer_result = load_model()
        
        # Log model type being used
        if config.settings.USE_CUSTOM_MODEL and verify_model_directories():
            logger.info("Successfully loaded custom model")
        else:
            logger.info(f"Using fallback model: {config.settings.MODEL_NAME}")
        
        # Welcome message
        welcome_message = "Hello, I'm your voice assistant. How can I help you today?"
        logger.info(welcome_message)
        speak(welcome_message)
        
        # Main loop
        logger.info("Entering main loop. Press Ctrl+C to exit.")
        while True:
            # Listen for voice input
            user_input = listen()
            
            # Process voice input if not empty
            if user_input:
                try:
                    # Generate response using AI model
                    response = generate_response(user_input)
                    
                    # Ensure we have a valid response
                    if response and isinstance(response, str):
                        # Speak the response - this will block until speech is complete
                        speak(response)
                        
                        # Only start cooldown after speech is complete
                        logger.info(f"Response complete. Cooling down for {COOLDOWN_TIME} seconds.")
                        time.sleep(COOLDOWN_TIME)
                    else:
                        logger.warning("Received invalid response from AI model.")
                        
                except Exception as e:
                    logger.error(f"Error during response generation or speech: {e}")
                    speak("I'm sorry, I encountered an error processing your request.")
                    time.sleep(COOLDOWN_TIME)
                
    except KeyboardInterrupt:
        logger.info("Received exit signal. Shutting down...")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
    finally:
        logger.info("Voice assistant terminated.")

if __name__ == "__main__":
    main()
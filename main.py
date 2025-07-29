"""
Main application for the desktop voice assistant.
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

def main():
    """Main function to run the voice assistant."""
    logger.info("Starting voice assistant...")
    
    try:
        # Load AI model at startup
        logger.info("Loading AI model...")
        load_model()
        
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

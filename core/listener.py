"""
Module to capture and recognize voice input.
"""

import speech_recognition as sr
from utils import logger
from config import settings

def listen():
    """
    Listen to microphone input and convert speech to text.
    
    Returns:
        str: Transcribed text or empty string if no speech detected
    """
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        logger.info("Listening...")
        
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        # Set pause threshold
        recognizer.pause_threshold = settings.PAUSE_THRESHOLD
        
        try:
            # Listen for user's voice input
            audio = recognizer.listen(source)
            logger.info("Processing speech...")
            
            # Use Google's speech recognition
            text = recognizer.recognize_google(audio, language=settings.LANGUAGE)
            logger.info(f"Heard: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.debug("Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Could not request results: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return ""

"""
Simple voice functionality without external dependencies.
"""

import sys

def listen():
    """Simple speech recognition function."""
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        with sr.Microphone() as source:
            print("🎤 Listening... (speak now)")
            
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for user's voice input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Use Google Speech Recognition
            text = recognizer.recognize_google(audio)
            return text
            
    except ImportError:
        print("speech_recognition not installed. Install with: pip install SpeechRecognition pyaudio")
        return ""
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return ""

def speak(text):
    """Simple text-to-speech function."""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Set properties
        engine.setProperty('rate', 150)    # Speed of speech
        engine.setProperty('volume', 0.9)  # Volume level
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
    except ImportError:
        print("pyttsx3 not installed. Install with: pip install pyttsx3")
    except Exception as e:
        print(f"Text-to-speech error: {e}")

if __name__ == "__main__":
    # Test the functions
    print("Testing voice functionality...")
    speak("Hello, this is a test of the text to speech system.")
    
    user_input = listen()
    if user_input:
        print(f"You said: {user_input}")
        speak(f"You said: {user_input}")
    else:
        print("No speech detected.")

"""
Main script to run the custom LLM as a conversational voice assistant.
Loads the model from models/custom_llm_improved and interacts with the user via voice and text.
"""

import os
import sys

# Import voice functionality
try:
    from simple_voice import listen, speak
    VOICE_AVAILABLE = True
    print("✅ Voice functionality available!")
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"❌ Voice functionality not available: {e}")
    print("💡 Install: pip install SpeechRecognition pyttsx3 pyaudio")

# Try to import model functionality
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    MODEL_LIBRARIES_AVAILABLE = True
except ImportError:
    MODEL_LIBRARIES_AVAILABLE = False
    print("Warning: transformers and torch not available. Model features limited.")

MODEL_DIR = "models/custom_llm_improved"

def load_model():
    """Load the custom LLM model."""
    if not MODEL_LIBRARIES_AVAILABLE:
        print("Model libraries not available. Using fallback responses.")
        return None, None
        
    print(f"Loading model from {MODEL_DIR} ...")
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"Model directory {MODEL_DIR} not found!")
            return None, None
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
        print("Model loaded successfully!\n")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def get_user_input(mode="text"):
    """Get input from user via text or voice."""
    if mode == "voice" and VOICE_AVAILABLE:
        print("🎤 Listening... (speak now)")
        user_input = listen()
        if user_input:
            print(f"You said: {user_input}")
            return user_input
        else:
            print("No speech detected, please try again or type your message:")
            return input("You: ").strip()
    else:
        return input("You: ").strip()

def respond_to_user(response, mode="text"):
    """Respond to user via text or voice."""
    print(f"Assistant: {response}")
    
    if mode == "voice" and VOICE_AVAILABLE:
        speak(response)

def generate_response(user_input, model, tokenizer):
    """Generate response using the model or fallback."""
    if model is None or tokenizer is None:
        # Fallback responses when model isn't available
        fallback_responses = {
            "hello": "Hello! I'm your voice assistant. How can I help you today?",
            "how are you": "I'm doing well, thank you for asking! How are you?",
            "what is your name": "I'm Echo, your voice assistant.",
            "what is france": "France is a country in Western Europe known for its culture, cuisine, and history.",
            "goodbye": "Goodbye! Have a great day!",
            "bye": "Goodbye! Take care!",
        }
        
        for key, response in fallback_responses.items():
            if key in user_input.lower():
                return response
        
        return "I'm sorry, I couldn't understand that. My full model isn't loaded right now."
    
    try:
        # Ensure tokenizer has pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        prompt = f"User: {user_input}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,  # Limit response length
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )
        
        # Decode only the new tokens (response part)
        response_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        # Clean up the response
        if "User:" in reply:
            reply = reply.split("User:")[0].strip()
        
        return reply if reply else "I'm not sure how to respond to that."
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while processing your request."

def chat(model, tokenizer):
    """Main chat loop with voice and text options."""
    print("\n🎙️ Echo Voice Assistant")
    print("========================")
    
    if VOICE_AVAILABLE:
        print("Voice mode available! Commands:")
        print("  - Type 'voice' to switch to voice input mode")
        print("  - Type 'text' to switch to text input mode")
        print("  - Say or type 'exit' or 'quit' to end")
    else:
        print("Text mode only (voice libraries not available)")
        print("Type 'exit' or 'quit' to end")
    
    print("-" * 40)
    
    mode = "text"  # Start in text mode
    
    while True:
        try:
            if mode == "voice" and VOICE_AVAILABLE:
                print(f"\n[Voice Mode] 🎤")
            else:
                print(f"\n[Text Mode] ⌨️")
            
            user_input = get_user_input(mode)
            
            if not user_input:
                continue
                
            # Check for mode switching commands
            if user_input.lower() == "voice" and VOICE_AVAILABLE:
                mode = "voice"
                respond_to_user("Switched to voice mode. You can now speak to me!", mode)
                continue
            elif user_input.lower() == "text":
                mode = "text"
                respond_to_user("Switched to text mode. You can now type to me!", mode)
                continue
            elif user_input.lower() in {"exit", "quit", "goodbye", "bye"}:
                farewell = "Goodbye! Thank you for using Echo Voice Assistant!"
                respond_to_user(farewell, mode)
                break
            
            # Generate and deliver response
            response = generate_response(user_input, model, tokenizer)
            respond_to_user(response, mode)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! (Interrupted by user)")
            break
        except Exception as e:
            print(f"Error in chat loop: {e}")
            continue

def main():
    """Main entry point."""
    print("🚀 Starting Echo Voice Assistant...")
    
    # Load the model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("⚠️  Model could not be loaded. Using fallback responses.")
        print("💡 To get full functionality, ensure transformers and torch are installed")
        print("   and the model files are in the correct location.")
    
    # Start the chat interface
    chat(model, tokenizer)

if __name__ == "__main__":
    main()

# Desktop Voice Assistant

A Python-based always-on desktop voice assistant for Windows that uses Hugging Face models to answer questions.

## Features

- Always-on voice listening (Jarvis-style)
- Uses Hugging Face's microsoft/phi-2 model for generating responses
- Speech-to-text using Google's Speech Recognition API
- Text-to-speech using pyttsx3
- Modular design for easy extension

## Installation

1. Clone the repository:

```
git clone <repository-url>
cd VoiceAssistant
```

2. Create and activate a virtual environment (recommended):

```
python -m venv venv
venv\Scripts\activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Run the main script:

```
python main.py
```

2. Speak to the assistant and wait for a response.
3. Press `Ctrl+C` to exit.

## Project Structure

```
voice_assistant/
├── main.py                 # Main loop for assistant
├── requirements.txt        # Python packages
│
├── config/
│   └── settings.py         # Voice, model, and general config
│
├── core/
│   ├── listener.py         # Uses microphone to capture speech
│   ├── responder.py        # Loads Hugging Face model and returns reply
│   ├── speaker.py          # Handles TTS (text-to-speech)
│
├── utils/
│   └── logger.py          # Print/log utility
│
└── assets/
    └── voices/            # (Optional) Voice presets or samples
```

## Configuration

You can modify the following settings in `config/settings.py`:

- AI model parameters (model name, max token length, temperature)
- Voice settings (rate, volume, gender)
- Listener settings (language, pause threshold)
- Application settings (cooldown time between responses)

## Requirements

- Python 3.7+
- PyAudio (may require additional system dependencies)
- Internet connection (for Google Speech Recognition)
- Sufficient RAM and disk space for the AI model

## Future Enhancements

- Calendar integration
- OS command execution
- Third-party API integration
- Custom wake word
- System tray integration

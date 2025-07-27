# 🤖 EchoAgents

The EchoAgents project aims to prove that true AI autonomy comes from a robust, stateful, and observable system architecture that orchestrates LLMs, rather than from the power of the language model alone.

---

## 🚀 Prerequisites

Before you begin, make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

To verify installations:

```bash
docker --version
docker-compose --version
```

---

## 🛠️ Setup & Run

### 1. Clone the Repository

```bash
git clone https://github.com/ArnavChat/EchoAgents.git
cd EchoAgents
```

### 2. Build the Docker Image

```bash
docker-compose build
```

> 💡 This installs all necessary Python packages listed in `requirements.txt`.

### 3. Run the App

```bash
docker-compose up
```

The app will now start inside a container and execute `test.py`. You should see output directly in your terminal.

---

## 📂 Project Structure

```
EchoAgents/
│
├── Dockerfile              # Builds the Docker image
├── docker-compose.yml      # Defines services and container behavior
├── requirements.txt        # Python dependencies
├── test.py                 # Entry-point script
└── (other Python modules and files)
```

---

## 🐳 Docker Commands (Quick Reference)

* **Rebuild the image from scratch:**

  ```bash
  docker-compose build --no-cache
  ```

* **Stop and remove running containers:**

  ```bash
  docker-compose down
  ```

* **Access the container shell (for debugging):**

  ```bash
  docker-compose run backend bash
  ```

---

## 🌱 Git Workflow (Pushing Changes)

Before pushing your changes, it’s recommended to create a separate branch:

### 1. Create and switch to a new branch:

```bash
git checkout -b feature/my-new-feature
```

### 2. Stage and commit your changes:

```bash
git add .
git commit -m "Add: My new feature"
```

### 3. Push to GitHub:

```bash
git push origin feature/my-new-feature
```

> 🔁 Then, create a pull request from that branch on GitHub to merge into `main`.

---

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

# Voice Assistant with Custom Language Model

This project implements a customizable voice assistant powered by a fine-tuned language model. The assistant can listen to voice commands, process them using a custom-trained language model, and respond with synthesized speech.

## Project Overview

The voice assistant system consists of three main components:

1. **Listener**: Captures audio input and converts speech to text
2. **Responder**: Processes text input using a language model and generates responses
3. **Speaker**: Converts text responses to spoken audio output

Additionally, the project includes tools for training and customizing your own language model.

## Architecture

![Voice Assistant Architecture](https://via.placeholder.com/800x400.png?text=Voice+Assistant+Architecture)

### Core Components

- **Main Application (`main.py`)**: Orchestrates the flow between listening, processing, and speaking
- **Core Modules**:
  - `listener.py`: Speech recognition using Google's API
  - `responder.py`: AI response generation using Hugging Face Transformers
  - `speaker.py`: Text-to-speech conversion using multiple fallback methods
- **Configuration**: Settings managed in `config/settings.py`
- **Utilities**: Logging, model management, and other helper functions

## Model Training and Customization

This project supports training custom language models using various datasets:

1. **Base Models**: Uses DistilGPT2 as the default base model, with options to use larger models
2. **Training Data**: 
   - Pre-made datasets like Alpaca (52,000+ examples)
   - Custom datasets that you create
   - Combined datasets for best results
3. **Training Process**:
   - Fine-tuning using Hugging Face's Transformers library
   - Customizable training parameters (epochs, batch size, learning rate)
   - GPU support for faster training

## Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework for model inference
- **Transformers**: Hugging Face library for transformer-based language models
- **SpeechRecognition**: Library for speech recognition
- **PyAudio**: Audio I/O library
- **pyttsx3**: Text-to-speech conversion library

### Training Libraries

- **Datasets**: Hugging Face dataset management library
- **Accelerate**: Library for distributed training
- **TensorBoard**: For training visualization and monitoring

## Project Structure

```
voice-assistant/
├── config/                  # Configuration files
│   └── settings.py          # Settings for the voice assistant
├── core/                    # Core functionality
│   ├── listener.py          # Speech-to-text module
│   ├── responder.py         # AI processing module
│   └── speaker.py           # Text-to-speech module
├── data/                    # Data for training models
│   └── training/            # Training datasets
│       ├── alpaca.json      # Alpaca dataset
│       └── custom_conversations.json  # Custom dataset
├── models/                  # Trained models
│   ├── custom_llm/          # First custom model
│   └── custom_llm_improved/ # Improved custom model
├── utils/                   # Utility functions
│   ├── logger.py            # Logging functionality
│   └── model_utils.py       # Model management utilities
├── docker-compose.yml       # Docker composition file
├── Dockerfile               # Docker container definition
├── main.py                  # Main application entry point
├── create_custom_model.py   # Script for creating custom models
├── combine_datasets.py      # Script for combining datasets
├── download_dataset.py      # Script for downloading datasets
├── requirements.txt         # Python dependencies
└── setup_training_environment.ps1  # Setup script for training
```

## Installation and Setup

### Method 1: Using Python Virtual Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/voice-assistant.git
   cd voice-assistant
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv voice_assistant_env
   .\voice_assistant_env\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the voice assistant**:
   ```powershell
   python main.py
   ```

### Method 2: Using Docker

1. **Build and run using Docker Compose**:
   ```bash
   docker-compose up
   ```

## Training Your Custom Model

### Quick Start

1. **Setup training environment**:
   ```powershell
   .\setup_training_environment.ps1
   ```

2. **Download or create a dataset**:
   ```powershell
   # Download existing dataset
   python download_dataset.py --dataset alpaca --output_dir data/training
   
   # Or use your custom dataset at data/training/custom_conversations.json
   ```

3. **Train your model**:
   ```powershell
   # Basic training
   python create_custom_model.py --dataset data/training/custom_conversations.json --epochs 3
   
   # Advanced training with more data and epochs
   python create_custom_model.py --dataset data/training/alpaca.json --epochs 10 --batch_size 4
   ```

4. **Update configuration to use your model**:
   Edit `config/settings.py` to set:
   ```python
   USE_CUSTOM_MODEL = True
   CUSTOM_MODEL_PATH = "models/custom_llm_improved"
   CUSTOM_TOKENIZER_PATH = "models/custom_llm_improved"
   ```

## Customizing Responses

You can improve your model's responses by:

1. **Creating custom datasets** with examples tailored to your use case
2. **Fine-tuning generation parameters** in `config/settings.py`:
   ```python
   MAX_LENGTH = 75         # Controls response length
   TEMPERATURE = 0.6       # Controls randomness (lower = more focused)
   TOP_P = 0.92            # Nucleus sampling parameter
   TOP_K = 50              # Limits vocabulary to top K tokens
   ```
3. **Adding a custom prompt template** in your model directory

## Performance Optimization

- **GPU Acceleration**: The system supports GPU acceleration if available
- **Optimized Inference**: Various parameters can be tuned for speed vs. quality tradeoffs
- **Docker Containerization**: Provides consistent environment and deployment

## Limitations and Future Improvements

1. **Model Quality**: The custom model quality depends on training data quality and quantity
2. **Speech Recognition**: Currently relies on Google's API, which requires internet connection
3. **Offline Operation**: Future versions aim to support fully offline operation
4. **Multi-turn Conversations**: Current implementation handles single-turn exchanges

## Troubleshooting

### Common Issues

1. **Microphone not working**: Check your microphone settings and permissions
2. **Model loading errors**: Ensure all model files are present in the specified directory
3. **Out of memory errors**: Reduce batch size or use a smaller model

### Logs

Logs are stored in the console output and can help diagnose issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers library
- Stanford Alpaca for the training dataset

# PowerShell script to set up a proper training environment for creating your own models

# Create a Python virtual environment
Write-Host "Creating a dedicated virtual environment for model training..." -ForegroundColor Green
python -m venv voice_assistant_env

# Activate the environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\voice_assistant_env\Scripts\Activate.ps1

# Update pip
Write-Host "Updating pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install PyTorch with CPU support (use CUDA version if you have a compatible GPU)
Write-Host "Installing PyTorch..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install pre-built wheels for common packages to avoid compilation issues
Write-Host "Installing pre-built wheels for training dependencies..." -ForegroundColor Green
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install pandas==2.0.3

# Install Hugging Face libraries
Write-Host "Installing Hugging Face libraries..." -ForegroundColor Green
pip install transformers==4.31.0
pip install datasets==2.14.0
pip install accelerate==0.21.0
pip install evaluate==0.4.0

# Install additional utilities
Write-Host "Installing additional utilities..." -ForegroundColor Green
pip install scikit-learn==1.3.0
pip install tensorboard==2.14.0
pip install tqdm==4.66.0
pip install colorama==0.4.6
pip install requests==2.31.0

# Install voice assistant specific packages
Write-Host "Installing voice assistant packages..." -ForegroundColor Green
pip install SpeechRecognition==3.10.0
pip install PyAudio==0.2.14
pip install pyttsx3==2.90

Write-Host "`nTraining environment setup complete!" -ForegroundColor Cyan
Write-Host "You can now proceed with training your custom model using:" -ForegroundColor Cyan
Write-Host "python train_custom_model.py --dataset data/training/your_dataset.json" -ForegroundColor Yellow

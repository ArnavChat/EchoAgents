FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for pip builds, including for PyAudio
# Install system dependencies required for pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    portaudio19-dev \     
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and wheel before installing dependencies
RUN pip install --upgrade pip wheel setuptools
RUN pip install torch==2.6.0+cpu torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "setup_custom_model.py"]

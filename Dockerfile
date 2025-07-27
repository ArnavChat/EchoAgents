FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and wheel before installing dependencies
RUN pip install --upgrade pip wheel setuptools

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "test.py"]

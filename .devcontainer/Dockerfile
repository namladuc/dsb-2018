FROM mcr.microsoft.com/devcontainers/javascript-node:20

RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    git \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv

# Activate venv automatically
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install ONNX tooling inside venv
RUN pip install \
    onnx \
    onnxruntime \
    onnxruntime-tools \
    polygraphy

# Create workspace
WORKDIR /workspace

# Node global tools
RUN npm install -g \
    vite \
    serve
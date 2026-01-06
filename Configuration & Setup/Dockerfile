# AI Image & Video Restoration - Docker Configuration
# Multi-stage build for optimized image size

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p output temp models/weights examples/input examples/output

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative commands:
# Run CLI: docker run -v $(pwd)/data:/app/data ai-restoration python restore.py --input /app/data/input.jpg --output /app/data/output.jpg
# Run examples: docker run ai-restoration python example_usage.py
# Interactive shell: docker run -it ai-restoration /bin/bash
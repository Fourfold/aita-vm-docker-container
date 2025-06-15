# Use NVIDIA CUDA base image from our private ECR (automatically managed by build script)
# The build script will pull from Docker Hub and push to private ECR if needed
ARG AWS_ACCOUNT_ID
ARG AWS_REGION=us-east-1
FROM ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/nvidia-cuda-base:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# CUDA and GPU optimization environment variables
ENV CUDA_LAUNCH_BLOCKING=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_CACHE_DISABLE=1
# Reduce memory fragmentation and improve L4 GPU compatibility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Configure timezone non-interactively
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python, Pip, and system dependencies for ML/document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    build-essential \
    pkg-config \
    swig \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for ML/document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1 \
    libfreetype6 \
    fonts-dejavu-core \
    fonts-liberation \
    ca-certificates \
    curl \
    wget \
    git \
    locales \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install newer GCC/G++ to fix PaddlePaddle C++ ABI compatibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc-11 \
    g++-11 \
    libstdc++6 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up locales for international text processing
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# Set Python 3.11 as default and install pip for Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3.11 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip setuptools wheel

# Create work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# PaddlePaddle installation
RUN pip3 install --no-cache-dir paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu124/
RUN pip3 install --no-cache-dir paddlex==3.0rc0 faiss-cpu==1.8.0 simsimd==1.1.2 --use-pep517

# Copy the rest of the files
COPY . .

# Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# Default command to run your app with better logging
CMD ["python3", "start_server.py"]
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

# Install poppler-utils
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean

# Set Python 3.11 as default and install pip for Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3.11 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip setuptools wheel

# Create work directory
WORKDIR /app

# Copy LoRA adapters first (can be cached if they don't change often)
COPY ./gemmax2_9b_finetuned /app/gemmax2_9b_finetuned

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

RUN apt-get update && apt-get install -y swig build-essential python3.11-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install faiss-cpu first with pre-built wheel, then other packages
RUN pip3 install --no-cache-dir paddlex==3.0rc0 faiss-cpu==1.8.0 simsimd==1.1.2 --use-pep517

# Platform detection for debugging
RUN echo "Platform detection:" && \
    uname -a && \
    python3 --version && \
    pip3 --version && \
    python3 -c "import platform; print(f'Platform: {platform.platform()}'); print(f'Machine: {platform.machine()}'); print(f'Architecture: {platform.architecture()}'); import sys; print(f'Python version: {sys.version}')"

# Install bitsandbytes with compatibility for glibc 2.31 (Amazon Linux 2)
RUN echo "Installing bitsandbytes with Python 3.11 compatibility..." && \
    # First try: install the latest available version
    (pip3 install --no-cache-dir --upgrade bitsandbytes && \
     echo "Successfully installed latest bitsandbytes") || \
    # Second try: install a specific newer version
    (pip3 install --no-cache-dir "bitsandbytes>=0.43.2" && \
     echo "Successfully installed bitsandbytes >= 0.43.2") || \
    # Fallback: install the working version
    (pip3 install --no-cache-dir "bitsandbytes==0.42.0" && \
     echo "Fallback: installed bitsandbytes 0.42.0") && \
    python3 -c "import bitsandbytes; print(f'bitsandbytes version: {bitsandbytes.__version__}')"

# Install other essential dependencies
RUN pip3 install --no-cache-dir torch==2.4.0 transformers==4.51.3 peft==0.15.2 accelerate==1.6.0

# Test bitsandbytes functionality
RUN python3 -c "import torch; from transformers import BitsAndBytesConfig; print('Testing BitsAndBytesConfig...'); bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16); print('BitsAndBytesConfig created successfully!'); print('CUDA available:', torch.cuda.is_available())"

# Install transformers to fix the version conflict
RUN pip3 install --no-cache-dir transformers==4.51.3

# Add GPU debugging and memory management for model initialization issues
RUN echo "GPU and CUDA debugging info:" && \
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count()) if torch.cuda.is_available()]; print(f'Current GPU: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" && \
    python3 -c "try: import paddlepaddle; print(f'PaddlePaddle version: {paddlepaddle.__version__}'); print('PaddlePaddle CUDA support:', paddlepaddle.is_compiled_with_cuda()); except Exception as e: print(f'PaddlePaddle error: {e}')" && \
    python3 -c "try: import paddlex; print(f'PaddleX version: {paddlex.__version__}'); except Exception as e: print(f'PaddleX error: {e}')"

# Test PaddleX model creation with timeout and memory constraints
RUN echo "Testing PaddleX model initialization..." && \
    timeout 300 python3 -c "\
import os; \
import signal; \
import sys; \
import torch; \
import gc; \
\
def timeout_handler(signum, frame): \
    print('Model initialization timed out after 5 minutes'); \
    sys.exit(1); \
\
signal.signal(signal.SIGALRM, timeout_handler); \
signal.alarm(300); \
\
try: \
    print('Setting memory management...'); \
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; \
    print('Importing PaddleX...'); \
    from paddlex import create_model; \
    print('Creating PP-DocLayout-L model (CPU mode)...'); \
    model = create_model(model_name='PP-DocLayout-L'); \
    print('Model created successfully!'); \
    print('Testing model prediction capability...'); \
    import numpy as np; \
    from PIL import Image; \
    dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)); \
    dummy_img.save('/tmp/test_img.png'); \
    output = model.predict('/tmp/test_img.png', batch_size=1); \
    print('Model prediction test successful!'); \
    print('Cleaning up...'); \
    del model; \
    gc.collect(); \
except Exception as e: \
    print(f'Model initialization error: {e}'); \
    import traceback; \
    traceback.print_exc(); \
    sys.exit(1); \
finally: \
    signal.alarm(0); \
\
print('PaddleX model test completed successfully!'); \
" || echo "Model initialization test failed or timed out"

# Copy the rest of the files
COPY . .

# Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# Default command to run your app with better logging
CMD ["python3", "start_server.py"]
# Use a base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
# ENV HF_TOKEN="your_hugging_face_token_here"
# # Or better, pass at build time or use secrets

# Install Python and Pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install poppler-utils
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Create work directory
WORKDIR /app

# Copy LoRA adapters first (can be cached if they don't change often)
COPY ./gemmax2_9b_finetuned /app/gemmax2_9b_finetuned

# Copy requirements and install dependencies
COPY requirements1.txt .
RUN pip3 install --no-cache-dir -r requirements1.txt
COPY requirements2.txt .
RUN pip3 install --no-cache-dir -r requirements2.txt

# Copy the rest of the files
COPY . .

# Expose the port FastAPI/Uvicorn will run on
EXPOSE 8000

# Default command to run your app with better logging
CMD ["python", "start_server.py"]
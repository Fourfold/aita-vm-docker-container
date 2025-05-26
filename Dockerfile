# Use official Python image from Docker Hub
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Make startup script executable
RUN chmod +x start_server.py

# Expose port 8000
EXPOSE 8000

# Default command to run your app with better logging
CMD ["python", "start_server.py"]
# Use official Python image
FROM python:3.9

# Prevent buffering issues in logs
ENV PYTHONUNBUFFERED=1  

# Expose port
EXPOSE 5002

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Gunicorn workers to 1 to prevent memory issues
CMD gunicorn --workers 1 --timeout 120 --bind 0.0.0.0:${PORT:-5002} app:app


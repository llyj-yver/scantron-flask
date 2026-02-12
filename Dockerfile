# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port for Render
EXPOSE 5000

# Start app with Gunicorn optimized for free tier (512MB RAM)
# --workers 1: Single worker to minimize memory usage
# --timeout 120: Allow 2 minutes for YOLO processing
# --max-requests 10: Restart worker after 10 requests to prevent memory leaks
# --worker-tmp-dir /dev/shm: Use RAM for worker heartbeat files
CMD ["gunicorn", "app:app", \
     "-b", "0.0.0.0:5000", \
     "--workers", "1", \
     "--timeout", "120", \
     "--max-requests", "10", \
     "--max-requests-jitter", "5", \
     "--worker-tmp-dir", "/dev/shm"]
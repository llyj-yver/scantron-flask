# Use Python 3.11 slim image (smaller footprint)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    YOLO_VERBOSE=False

# Install system dependencies for OpenCV
# Fixed: Using correct package names for Debian/Ubuntu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results && \
    chmod 755 uploads results

# Expose port
EXPOSE 5000

# Start with optimized Gunicorn settings for Render free tier
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--worker-class", "sync", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keepalive", "5", \
     "--max-requests", "50", \
     "--max-requests-jitter", "10", \
     "--worker-tmp-dir", "/dev/shm", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
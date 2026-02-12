# Use Python 3.11 slim image (smaller footprint)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # PyTorch/YOLO optimizations
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    # Reduce YOLO verbosity
    YOLO_VERBOSE=False

# Install system dependencies (minimal set for OpenCV headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache
    rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results && \
    chmod 755 uploads results

# Expose port
EXPOSE 5000

# Health check (optional but recommended for Render)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/test', timeout=5)"

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
     "--error-logfile", "-", \
     "--capture-output"]
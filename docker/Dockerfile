# Multi-stage build for smaller image size
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
# (gcc is good to keep for building certain python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- NEW: Download NLTK Data ---
# We download the data to a specific folder in the builder stage
RUN python -m nltk.downloader -d /app/nltk_data punkt punkt_tab


# Final stage
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/var/exports /app/var/chroma_db && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# --- NEW: Copy NLTK Data from builder ---
# We place it in the user's home directory where NLTK looks by default
COPY --from=builder --chown=appuser:appuser /app/nltk_data /home/appuser/nltk_data

# Copy application code
COPY --chown=appuser:appuser app ./app
# Only copy local var contents if you have default config files there
#COPY --chown=appuser:appuser var ./var

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    # Explicitly tell NLTK where to look (just to be safe)
    NLTK_DATA=/home/appuser/nltk_data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
# (Uses 'requests' which is in your requirements.txt)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

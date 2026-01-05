# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies using UV
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY static/ ./static/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Use a lightweight Python image
FROM python:3.10-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch transformers fastapi uvicorn

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . .

# Expose the port for the API
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]

# Multi-stage Dockerfile for Agentic Knee MRI System
# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python Backend + Static Frontend
FROM python:3.11-slim

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies (curl for healthchecks, libgl for OpenCV if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source, models, and RAG data
COPY backend/app ./app
COPY backend/model ./model
COPY backend/sample_data ./sample_data
COPY backend/rag_data ./rag_data

# Copy built frontend from Stage 1 into the location expected by FastAPI
COPY --from=frontend-builder /frontend/dist /frontend/dist

# Expose port (7860 is default for Hugging Face Spaces Docker)
EXPOSE 7860

# Run FastAPI backend with Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

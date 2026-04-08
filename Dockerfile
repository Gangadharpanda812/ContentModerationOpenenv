FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY content_moderation_openenv/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenEnv and other dependencies
RUN pip install --no-cache-dir \
    openenv \
    pydantic \
    fastapi \
    uvicorn \
    openai

# Copy the entire project
COPY . /app/

# Expose port for the web server
EXPOSE 8000

# Default command to run the server
CMD ["uvicorn", "content_moderation_openenv.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
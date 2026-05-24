FROM python:3.12-slim

WORKDIR /app

# Install Node.js (required for npx medical-mcp)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# CACHE_BUST: increment to force full reinstall (e.g. when adding new deps)
ARG CACHE_BUST=3

# Install deps first (layer cache — only rebuilds when requirements change)
COPY clinical_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source + artefacts (pkl files included via .dockerignore)
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn clinical_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

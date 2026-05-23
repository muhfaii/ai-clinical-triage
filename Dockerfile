FROM python:3.12-slim

WORKDIR /app

# Install deps first (layer cache — only rebuilds when requirements change)
COPY clinical_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source + artefacts (pkl files included via .dockerignore)
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn clinical_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

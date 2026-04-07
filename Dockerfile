FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
	CMD sh -c 'curl -f http://localhost:${PORT:-7860}/health || exit 1'
CMD ["sh", "-c", "uvicorn space_app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]

# --------------------------------------------------------------------------
# RegComplianceEnv — GDPR compliance checker (OpenEnv hackathon submission)
# Runs within 8 GB RAM / 2 vCPU — text processing + API calls only
# --------------------------------------------------------------------------

FROM python:3.11-slim

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install Python dependencies (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy all project files
COPY . .

# 3. Validate static GDPR cache exists and is valid JSON (build-time check)
RUN python -c "import json; data = json.load(open('data/gdpr_articles.json')); assert len(data) >= 5, 'GDPR articles incomplete'; print('GDPR cache OK:', list(data.keys()))"

# 4. Expose HuggingFace Spaces default port
EXPOSE 7860

# 5. Health check — judges poll /health during evaluation
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 6. Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

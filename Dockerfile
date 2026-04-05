# --------------------------------------------------------------------------
# RegComplianceEnv — Lightweight GDPR compliance checker
# Runs within 8 GB RAM (text processing + API calls only, no ML models)
# --------------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# 1. Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy all project files
COPY . .

# 3. Ensure data directories exist
RUN mkdir -p data/sample_policies

# 4. Pre-build GDPR cache at build time (falls back to static data on failure)
ARG FIRECRAWL_API_KEY=""
ENV FIRECRAWL_API_KEY=$FIRECRAWL_API_KEY
RUN python scraper.py || echo "Scraper failed — will use static GDPR fallback at runtime"

# 5. Expose HF Spaces default port
EXPOSE 7860

# 6. Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

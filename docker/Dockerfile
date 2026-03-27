# ─────────────────────────────────────────────────────────────────
# Dockerfile — API Prédiction fin d'orage
# ─────────────────────────────────────────────────────────────────
#


FROM python:3.11-slim

# 2. On crée un dossier de travail dans la boîte
WORKDIR /app


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY modele_fin_orage.json .
COPY label_encoder_airport.pkl .
COPY feature_cols.pkl .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
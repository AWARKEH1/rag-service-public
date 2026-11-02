FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir -r requirements.txt

# download the embedding model at build time so the container starts fast
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"

COPY rag_sp ./rag_sp
RUN useradd -m app && chown -R app /app
USER app

# Cloud Run and HF Spaces pass the port in $PORT
CMD ["sh", "-c", "uvicorn rag_sp.api:app --host 0.0.0.0 --port ${PORT}"]

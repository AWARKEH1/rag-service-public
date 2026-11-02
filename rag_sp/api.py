"""FastAPI service.

    uvicorn rag_sp.api:app --port 8000
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{"question": "..."}'
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .pipeline import RAG

log = logging.getLogger("rag_sp")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["rag"] = RAG()  # load the embedding model and open Qdrant once
    yield


app = FastAPI(title="Assistant Service-Public", version="0.1.0", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    refs: list[int]
    title: str
    url: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    try:
        res = state["rag"].ask(q.question)
    except Exception as e:  # LLM or vector-store failure: do not leak internals to the client
        log.exception("ask failed")
        raise HTTPException(status_code=503, detail="Service momentanément indisponible") from e
    log.info("question=%r latency=%.2fs sources=%d", q.question, res["latency_s"], len(res["sources"]))
    return res


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

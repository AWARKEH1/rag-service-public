"""Retrieve (dense search + cross-encoder reranking) -> generate. Shared by the API and the evaluation."""

import os
import time
from typing import Dict, List

from .config import Settings, get_settings
from .generate import NO_CONTEXT, build_messages, chat, cited_sources, normalize_citations
from .store import Embedder, get_client, search

# Passages scoring below this (after reranking) are dropped; if none is left the API answers "not found"
# without calling the LLM. Unset = disabled. Calibrate it with eval/retrieval.py on out-of-scope questions.
MIN_SCORE = os.environ.get("MIN_SCORE") or None


def passage_text(p: Dict) -> str:
    return f"{p['title']} > {p['heading']}\n{p['text']}" if p.get("heading") else f"{p['title']}\n{p['text']}"


class Reranker:
    """A cross-encoder reads the question and the passage together, so it judges relevance much better
    than comparing two separately computed vectors - but it is too slow to run on the whole corpus.
    Hence: fast vector search for ~30 candidates, then the cross-encoder re-orders them."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, question: str, passages: List[Dict], k: int) -> List[Dict]:
        if not passages:
            return []
        scores = self.model.predict([(question, passage_text(p)) for p in passages], batch_size=16)
        ranked = sorted(zip(passages, scores), key=lambda x: -x[1])
        return [{**p, "vector_score": p["score"], "score": float(s)} for p, s in ranked[:k]]


class RAG:
    def __init__(self, settings: Settings = None):
        self.s = settings or get_settings()
        self.emb = Embedder(self.s.embedding_model)
        self.client = get_client(self.s)
        self.reranker = Reranker(self.s.rerank_model) if self.s.rerank_model else None

    def retrieve(self, question: str, k: int = None, rerank: bool = True) -> List[Dict]:
        k = k or self.s.top_k
        if rerank and self.reranker:
            cands = search(self.client, self.s.collection, self.emb.query(question), max(k, self.s.rerank_candidates))
            return self.reranker.rerank(question, cands, k)
        return search(self.client, self.s.collection, self.emb.query(question), k)

    def ask(self, question: str) -> Dict:
        t0 = time.time()
        passages = self.retrieve(question)
        if MIN_SCORE is not None:
            passages = [p for p in passages if p["score"] >= float(MIN_SCORE)]
        if not passages:
            return {"answer": NO_CONTEXT, "sources": [], "passages": [], "latency_s": round(time.time() - t0, 2)}
        answer = normalize_citations(chat(build_messages(question, passages), self.s))
        return {"answer": answer, "sources": cited_sources(answer, passages), "passages": passages,
                "latency_s": round(time.time() - t0, 2)}

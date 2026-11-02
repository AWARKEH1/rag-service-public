"""Embeddings (multilingual E5) + Qdrant vector store. Heavy imports stay inside functions so the
pure-Python parts of the project can be tested without them."""

import uuid
from typing import Dict, List

from .chunk import Chunk
from .config import Settings


class Embedder:
    """E5 models expect the prefixes 'query: ' and 'passage: '."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()

    def _encode(self, texts: List[str], prefix: str) -> List[List[float]]:
        texts = [prefix + t for t in texts] if self.e5 else texts
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()

    def passages(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, "passage: ")

    def query(self, text: str) -> List[float]:
        return self._encode([text], "query: ")[0]

    @property
    def dim(self) -> int:
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return get_dim()


def get_client(s: Settings):
    from qdrant_client import QdrantClient
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
    return QdrantClient(path=s.qdrant_path)


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be int or UUID: derive a stable UUID from the chunk id (re-indexing overwrites)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def payload(c: Chunk) -> Dict[str, str]:
    return {"chunk_id": c.id, "fiche_id": c.fiche_id, "title": c.title, "url": c.url,
            "heading": c.heading, "text": c.text}


def recreate_collection(client, name: str, dim: int) -> None:
    from qdrant_client.models import Distance, VectorParams
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))


def upsert(client, name: str, chunks: List[Chunk], vectors: List[List[float]]) -> None:
    from qdrant_client.models import PointStruct
    client.upsert(name, points=[PointStruct(id=point_id(c.id), vector=v, payload=payload(c))
                                for c, v in zip(chunks, vectors)])


def search(client, name: str, vector: List[float], k: int) -> List[Dict]:
    hits = client.query_points(name, query=vector, limit=k, with_payload=True).points
    return [{**h.payload, "score": float(h.score)} for h in hits]

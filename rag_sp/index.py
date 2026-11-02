"""Parse -> chunk -> embed -> Qdrant.

    python -m rag_sp.index                 # all fiches in data/raw
    python -m rag_sp.index --limit 200     # quick test run
"""

import argparse
import itertools
import json
import os
import time

from .chunk import chunk_all
from .config import get_settings
from .parse import iter_fiches
from .store import Embedder, get_client, recreate_collection, upsert


def batched(it, n):
    it = iter(it)
    while batch := list(itertools.islice(it, n)):
        yield batch


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(s.data_dir, "raw"))
    ap.add_argument("--limit", type=int, default=0, help="index only the first N fiches")
    args = ap.parse_args()

    fiches = list(iter_fiches(args.raw))
    if args.limit:
        fiches = fiches[:args.limit]
    # keep a light copy of the corpus: used to build the evaluation set
    os.makedirs(s.data_dir, exist_ok=True)
    with open(os.path.join(s.data_dir, "fiches.jsonl"), "w", encoding="utf-8") as f:
        for fi in fiches:
            f.write(json.dumps({"id": fi.id, "title": fi.title, "url": fi.url, "text": fi.text},
                               ensure_ascii=False) + "\n")

    emb, client = Embedder(s.embedding_model), get_client(s)
    recreate_collection(client, s.collection, emb.dim)
    t0, n = time.time(), 0
    for batch in batched(chunk_all(fiches, s.chunk_words, s.chunk_overlap), 256):
        upsert(client, s.collection, batch, emb.passages([c.for_embedding() for c in batch]))
        n += len(batch)
    print(f"{len(fiches)} fiches, {n} chunks indexed in '{s.collection}' ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

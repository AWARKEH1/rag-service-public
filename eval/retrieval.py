"""Retrieval evaluation: does the expected fiche come back in the top-k passages?

    python -m eval.retrieval --questions eval/questions.jsonl --k 10
"""

import argparse
import json

from rag_sp.pipeline import RAG

from .metrics import first_rank, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-rerank", action="store_true", help="vector search only (baseline)")
    ap.add_argument("--out", default="eval/results_retrieval.jsonl")
    args = ap.parse_args()

    rag = RAG()
    qs = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    ranks = []
    with open(args.out, "w", encoding="utf-8") as out:
        for q in qs:
            hits = rag.retrieve(q["question"], args.k, rerank=not args.no_rerank)
            r = first_rank(q["fiche_id"], [h["fiche_id"] for h in hits])
            ranks.append(r)
            out.write(json.dumps({**q, "rank": r, "top_scores": [round(h["score"], 3) for h in hits[:3]],
                                  "retrieved": [h["fiche_id"] for h in hits]}, ensure_ascii=False) + "\n")
    mode = "vector search only" if args.no_rerank else "vector search + reranker"
    print(mode, json.dumps({"n": len(qs), **summary(ranks)}, indent=2))


if __name__ == "__main__":
    main()

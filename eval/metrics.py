"""Retrieval metrics (pure Python)."""

from typing import Dict, List, Sequence


def first_rank(expected: str, retrieved: Sequence[str]) -> int:
    """1-based rank of the first passage from the expected fiche, 0 if absent."""
    for i, fid in enumerate(retrieved, 1):
        if fid == expected:
            return i
    return 0


def hit_rate(ranks: List[int], k: int) -> float:
    return sum(1 for r in ranks if 0 < r <= k) / max(1, len(ranks))


def mrr(ranks: List[int]) -> float:
    return sum(1 / r for r in ranks if r) / max(1, len(ranks))


def summary(ranks: List[int], ks=(1, 3, 5)) -> Dict[str, float]:
    out = {f"hit@{k}": round(hit_rate(ranks, k), 3) for k in ks}
    out["mrr"] = round(mrr(ranks), 3)
    return out

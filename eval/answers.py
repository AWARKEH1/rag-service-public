"""End-to-end evaluation with an LLM judge: for each question, is the answer correct (compared with
the reference answer) and grounded (every claim supported by the retrieved passages)?

    python -m eval.answers --limit 40          # resumable: re-run to continue after a rate limit or a crash

The judge is a different model from the one that writes the answers (JUDGE_MODEL, default
openai/gpt-oss-20b), so a model does not grade its own output and the two use separate free-tier quotas.
Latencies measured on a free tier include rate-limit waits: treat them as upper bounds.
"""

import argparse
import dataclasses
import json
import os
import random
import re

from rag_sp.config import get_settings
from rag_sp.generate import chat, format_context, normalize_citations
from rag_sp.pipeline import RAG

JUDGE = (
    "Tu évalues un assistant. Question : {q}\nRéponse de référence : {ref}\n\nExtraits fournis à "
    "l'assistant :\n{ctx}\n\nRéponse de l'assistant : {ans}\n\nRéponds uniquement en JSON : "
    "{{\"correct\": true/false (la réponse contient-elle l'information de la référence, sans la contredire ?), "
    "\"grounded\": true/false (chaque affirmation est-elle appuyée par les extraits ?)}}"
)
KEYS = ("correct", "grounded")  # judged by the LLM


def cites(prediction: str) -> bool:
    """Checked in code, not by the judge: does the answer cite at least one passage as [n]?"""
    return bool(re.search(r"\[\d+\]", normalize_citations(prediction)))


def summarize(rows):
    n = max(1, len(rows))
    lat = sorted(r["latency_s"] for r in rows)
    out = {k: round(sum(bool(r["judge"].get(k)) for r in rows) / n, 3) for k in KEYS}
    out["cites"] = round(sum(cites(r["prediction"]) for r in rows) / n, 3)
    return {"n": len(rows), **out, "latency_p50_s": lat[len(lat) // 2] if lat else None}


def main() -> None:
    s = get_settings()
    judge = dataclasses.replace(s, llm_model=os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="evaluate a fixed random sample of N questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results_answers.jsonl")
    args = ap.parse_args()

    qs = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    if args.limit:
        qs = random.Random(args.seed).sample(qs, min(args.limit, len(qs)))
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                done[row["question"]] = row

    rag = RAG()
    with open(args.out, "a", encoding="utf-8") as out:
        for i, q in enumerate(qs, 1):
            if q["question"] in done:
                continue
            res = rag.ask(q["question"])
            raw = chat([{"role": "user", "content": JUDGE.format(
                q=q["question"], ref=q["answer"], ctx=format_context(res["passages"]), ans=res["answer"])}],
                judge, temperature=0)
            try:
                v = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            except ValueError:
                v = {}
            row = {**q, "prediction": res["answer"], "judge": v, "latency_s": res["latency_s"]}
            done[q["question"]] = row
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 5 == 0:
                print(f"{i} / {len(qs)}", flush=True)
    rows = [done[q["question"]] for q in qs if q["question"] in done]
    print(json.dumps({"answer_model": s.llm_model, "judge_model": judge.llm_model, **summarize(rows)}, indent=2))


if __name__ == "__main__":
    main()

"""Draft evaluation questions from random fiches with the LLM. Each draft keeps the fiche it came
from, which becomes the expected source. The drafts MUST be reviewed by hand (fix, delete, rephrase
like a real user) before being used: a generated test set is only as good as its review.

    python -m eval.make_questions --n 120 --out eval/questions_draft.jsonl
"""

import argparse
import json
import random
import os

from rag_sp.config import get_settings
from rag_sp.generate import chat

PROMPT = (
    "Voici une fiche de Service-Public.fr. Écris UNE question qu'un usager pourrait poser et à laquelle "
    "cette fiche répond, formulée avec ses propres mots (sans reprendre le titre), puis la réponse courte "
    "exacte d'après la fiche. Réponds en JSON : {{\"question\": \"...\", \"answer\": \"...\"}}\n\n"
    "Titre : {title}\n\n{text}"
)


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiches", default=f"{s.data_dir}/fiches.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/questions_draft.jsonl")
    args = ap.parse_args()

    fiches = [json.loads(line) for line in open(args.fiches, encoding="utf-8")]
    random.Random(args.seed).shuffle(fiches)
    # resumable: fiches already in the output file are skipped, new drafts are appended
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(line)["fiche_id"] for line in open(args.out, encoding="utf-8") if line.strip()}
    with open(args.out, "a", encoding="utf-8") as out:
        for i, f in enumerate(fiches[:args.n], 1):
            if f["id"] in done:
                continue
            raw = chat([{"role": "user", "content": PROMPT.format(title=f["title"], text=f["text"][:3500])}],
                       s, temperature=0.7)
            try:
                qa = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            except ValueError:
                continue
            out.write(json.dumps({"question": qa["question"], "answer": qa["answer"], "fiche_id": f["id"],
                                  "title": f["title"]}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 10 == 0:
                print(f"{i} / {args.n}", flush=True)
    print(f"drafts written to {args.out} -> review them, then save as eval/questions.jsonl")


if __name__ == "__main__":
    main()

# Assistant Service-Public — RAG on French administrative guides

> Ask a question about a French administrative procedure (ID card, housing benefits, unemployment, taxes…) and get a short answer **with numbered sources** taken from the official Service-Public.fr guides — or an honest "I don't know".

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-API-009688) ![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-dc244c) ![Docker](https://img.shields.io/badge/Docker-ready-2496ed) ![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)

## Why this project

Service-Public.fr publishes several thousand guides ("fiches pratiques"). They are reliable but long, and users ask questions in their own words, not with the guide's title. This project builds a retrieval-augmented generation (RAG) service on top of them, with the parts a real deployment needs: structure-aware parsing, a vector database, an API, a container, CI and — most important — **an evaluation set to measure whether changes actually help**.

## Architecture

```
data.gouv.fr (DILA XML) ──► parse ──► section-aware chunks ──► multilingual E5 embeddings ──► Qdrant
                                                                                               │
user question ──► FastAPI /ask ──► embed query ──► top-k passages ─────────────────────────────┘
                                         │
                                         └──► LLM (answer only from passages, cite [n]) ──► answer + sources
```

| Step | Choice | Why |
|---|---|---|
| Parsing | `rag_sp/parse.py` walks the DILA XML (chapters, sub-chapters, paragraphs, lists) and drops navigation blocks | keeps the document structure instead of flattening HTML |
| Chunking | ≤ 300 words, 50 overlap, never across two sections, each chunk prefixed with *title > chapter > sub-chapter* | a retrieved chunk is understandable on its own |
| Embeddings | `intfloat/multilingual-e5-small` (open, runs on CPU) with `query:` / `passage:` prefixes | good French retrieval at low cost |
| Vector DB | Qdrant (local file mode, Docker, or Qdrant Cloud) | same code from laptop to deployment |
| Generation | any OpenAI-compatible API (default: gpt-oss-120b (open weights, Apache 2.0) on Groq's free tier; Mistral works by changing two variables), temperature 0.1, strict prompt: answer only from the passages, cite `[n]`, say when it doesn't know | grounded, checkable answers |
| Serving | FastAPI (`/ask`, `/health`, a minimal web page), Docker, `$PORT` for Cloud Run / HF Spaces | |

## Evaluation

The test set is built from the corpus itself: `eval/make_questions.py` drafted one question per random guide with an LLM (120 drafts), then **every question was checked against its source guide**: 6 removed (vague or truncated reference answers), 9 corrected (one draft was in English, an ambiguous acronym, a wrong premise, incomplete answers). **114 questions** remain in [`eval/questions.jsonl`](eval/questions.jsonl); the guide each question came from is the expected source.

| Script | Measures |
|---|---|
| `eval/retrieval.py` | hit@1, hit@3, hit@5, MRR — is the expected guide in the retrieved passages? |
| `eval/answers.py` | LLM-as-judge: correct vs reference, grounded in the passages, cites a source; latency p50 / p95 |

### Retrieval (114 questions, 2,903 guides, 20,119 chunks)

| | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Vector search only (multilingual-e5-small) | 66.7 % | 83.3 % | 86.0 % | 0.754 |
| **+ cross-encoder reranking** (bge-reranker-v2-m3, top 30) | **85.1 %** | **93.9 %** | **93.9 %** | **0.894** |

Reranking puts the right guide first for **85 %** of questions instead of 67 %. It was added after a failure analysis: for *"Combien de temps à l'avance je dois prévenir mon propriétaire pour partir ?"*, the two relevant guides (notice period for tenants) were only 14th and 18th with vector search — present, but too low.

How to read these numbers: questions were generated from one guide each, so they may share wording with it (optimistic); but only that one guide counts as correct, even when another guide also answers the question (pessimistic).

### Answers (random sample of 40 questions)

Answers by `gpt-oss-120b`; correctness and grounding judged by a **different** model (`gpt-oss-20b`) so no model grades itself; citations checked in code. Every judgement was then **read by hand**, and one judge error was corrected (see below). Raw outputs: [`eval/results_answers.jsonl`](eval/results_answers.jsonl).

| | Result |
|---|---|
| Correct (agrees with the reference answer) | **38 / 40** (95 %) |
| Grounded (every claim supported by the retrieved passages) | **39 / 40** (97.5 %) |
| Cites at least one source `[n]` | **38 / 40** — the 2 others are the 2 "not found" answers below |
| Median latency (retrieval + reranking + generation) | 4.3 s |

**The two failures are both retrieval misses**, not generation errors:
- *Legal interest on a 2,000 € debt*: the passage with the worked example was not retrieved; the assistant said it could not answer from the extracts — the intended behaviour.
- *First technical inspection of a car*: passages about motorbikes, vans and vintage cars were retrieved instead of the car guide. The assistant said so, **but then added a rule from its own knowledge** ("4 years, 2-month margin"), which is not in the extracts and is inaccurate (the guide says within the 6 months before the 4th anniversary). The judge marked this answer *correct* — I overrode it to incorrect. The prompt now forbids adding anything not in the extracts.

**What I learned from reading the outputs** (not visible in the averages):
- `gpt-oss` sometimes cites in its native format `【1†L9-L12】` instead of `[1]` (4 / 18 answers in a first run): the sources were silently missing from the UI. Citations are now normalised, and counted in code — the LLM judge had also mis-scored them.
- An LLM judge is lenient on partially wrong answers; reported numbers are after manual review.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your LLM_API_KEY (never commit .env)

python -m rag_sp.download       # ~25 MB from data.gouv.fr -> data/raw/
python -m rag_sp.index          # parse, chunk, embed, store in Qdrant (add --limit 200 for a quick try)
uvicorn rag_sp.api:app --port 8000   # http://localhost:8000
```

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "Combien de temps est valable une carte d identite ?"}'
```

With Docker: `docker compose up --build` (Qdrant + API), then run `python -m rag_sp.index` with `QDRANT_URL=http://localhost:6333`.

Evaluation:

```bash
python -m eval.make_questions --n 120     # drafts -> review by hand -> eval/questions.jsonl
python -m eval.retrieval --no-rerank --out eval/results_retrieval_baseline.jsonl
python -m eval.retrieval
python -m eval.answers
```

Tests: `pytest` (no network or model needed).

## Repository

```
rag_sp/
├── config.py      # settings from environment variables
├── download.py    # fetch and unpack the DILA dataset
├── parse.py       # DILA XML -> fiches with titled sections
├── chunk.py       # section-aware chunking with context prefix
├── store.py       # E5 embeddings + Qdrant
├── index.py       # parse -> chunk -> embed -> upsert
├── generate.py    # prompt, LLM call (OpenAI-compatible), cited sources
├── pipeline.py    # retrieve -> generate
├── api.py         # FastAPI service
└── static/        # minimal web page
eval/              # test-set drafting, retrieval and answer evaluation
tests/             # unit tests
```

## Data and licence

- Data: **Service-Public.gouv.fr / DILA** — *Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers*, published on [data.gouv.fr](https://www.data.gouv.fr/) under the [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/). The data is downloaded at run time and not stored in this repository.
- This is a personal project, not affiliated with the French administration. Answers are generated automatically and may be wrong: always check the official guide linked in the sources.
- Code: MIT. The pipeline design was inspired by the free [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) course (DataTalks.Club); the code here was written for this project.

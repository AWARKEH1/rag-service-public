"""Prompt building and answer generation through any OpenAI-compatible chat API (key read from LLM_API_KEY)."""

import json
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions sur les démarches administratives en France. "
    "Réponds uniquement à partir des extraits fournis, en français clair et concis. "
    "Cite tes sources avec leur numéro entre crochets droits, par exemple [1] ou [2][3], et uniquement sous "
    "cette forme. "
    "Si les extraits ne permettent pas de répondre, dis-le simplement : n'ajoute aucune information qui "
    "ne figure pas dans les extraits, même si tu crois la connaître. "
    "Ces informations sont générales : invite l'utilisateur à vérifier sur service-public.fr pour son cas."
)
NO_CONTEXT = "Je n'ai pas trouvé d'information pertinente dans les fiches Service-Public.fr pour cette question."


def format_context(passages: List[Dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        where = f"{p['title']} > {p['heading']}" if p.get("heading") else p["title"]
        blocks.append(f"[{i}] {where}\n{p['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, passages: List[Dict]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extraits :\n\n{format_context(passages)}\n\nQuestion : {question}"}]


def normalize_citations(answer: str) -> str:
    """Some models (e.g. gpt-oss) cite in their native format "【1】" or "【1†L9-L12】" despite the prompt:
    rewrite those as "[1]" so the sources are shown and counted."""
    return re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", answer)


def cited_sources(answer: str, passages: List[Dict]) -> List[Dict]:
    """Sources actually cited in the answer, one per fiche (in order of first citation), with every passage
    number that points to it - so "[3]" is still explained when passages 1 and 3 come from the same fiche."""
    by_fiche = {}
    for n in re.findall(r"\[(\d+)\]", answer):
        i = int(n) - 1
        if 0 <= i < len(passages):
            src = by_fiche.setdefault(passages[i]["fiche_id"], {"refs": [], "title": passages[i]["title"],
                                                                 "url": passages[i]["url"]})
            if i + 1 not in src["refs"]:
                src["refs"].append(i + 1)
    return list(by_fiche.values())


def retry_delay(message: str, headers=None, default: float = 15.0) -> float:
    """Seconds to wait after a 429, from the Retry-After header or 'try again in 12.15s' in the message."""
    if headers is not None and headers.get("retry-after"):
        try:
            return float(headers.get("retry-after")) + 0.5
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5) if m else default


def chat(messages: List[Dict[str, str]], settings, temperature: float = 0.1, timeout: int = 60,
         max_retries: int = 6) -> str:
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set")
    body = json.dumps({"model": settings.llm_model, "messages": messages, "temperature": temperature}).encode()
    req = urllib.request.Request(settings.llm_base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {settings.llm_api_key}",
                                          "Content-Type": "application/json",
                                          "User-Agent": "rag-service-public"})
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code == 429 and attempt < max_retries:  # free-tier rate limit: wait as told, then retry
                time.sleep(retry_delay(msg, e.headers))
                continue
            raise RuntimeError(f"LLM API error {e.code}: {msg}") from e  # unknown model, bad key...
    raise RuntimeError("unreachable")

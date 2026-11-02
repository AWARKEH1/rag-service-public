"""Section-aware chunking.

Service-Public fiches are split into many short sections (one per situation / case, often 30-80
words). One chunk per section gives thousands of tiny chunks that lack context, so consecutive
sections are packed together up to `size` words. The chunk's heading is the part of the path the
packed sections share (e.g. "En France > Préparer les documents"); the remaining sub-headings are
kept inline as "[Vous avez un passeport valide]" so no information is lost. A section longer than
`size` is split with overlap. Every chunk is prefixed with the fiche title and heading when embedded,
so a retrieved chunk still makes sense on its own.
"""

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

from .parse import Fiche


@dataclass
class Chunk:
    id: str
    fiche_id: str
    title: str
    url: str
    heading: str
    text: str

    def for_embedding(self) -> str:
        return f"{self.title} > {self.heading}\n{self.text}" if self.heading else f"{self.title}\n{self.text}"


def split_words(text: str, size: int, overlap: int) -> List[str]:
    if size <= overlap:
        raise ValueError("chunk size must be larger than the overlap")
    words = text.split(" ")
    if len(words) <= size:
        return [text]
    out, start = [], 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def common_prefix(paths: List[List[str]]) -> List[str]:
    out = []
    for parts in zip(*paths):
        if any(p != parts[0] for p in parts):
            break
        out.append(parts[0])
    return out


def render(group: List[Tuple[List[str], str]]) -> Tuple[List[str], str]:
    """Heading shared by the group + text with the remaining sub-headings inline."""
    prefix = common_prefix([h for h, _ in group])
    lines, last = [], None
    for heading, text in group:
        rest = " > ".join(heading[len(prefix):])
        if rest and rest != last:
            lines.append(f"[{rest}]")
        last = rest
        lines.append(text)
    return prefix, "\n".join(lines)


def chunk_fiche(fiche: Fiche, size: int = 300, overlap: int = 50) -> Iterator[Chunk]:
    n_words = lambda t: len(t.split())  # noqa: E731
    pieces = [(sec.heading, p) for sec in fiche.sections for p in split_words(sec.text, size, overlap)]
    groups, cur, cur_words = [], [], 0
    for heading, text in pieces:
        w = n_words(text) + len(heading)
        if cur and cur_words + w > size:
            groups.append(cur)
            cur, cur_words = [], 0
        cur.append((heading, text))
        cur_words += w
    if cur:
        groups.append(cur)
    for n, group in enumerate(groups):
        heading, text = render(group)
        yield Chunk(f"{fiche.id}-{n}", fiche.id, fiche.title, fiche.url, " > ".join(h for h in heading if h), text)


def chunk_all(fiches: Iterable[Fiche], size: int = 300, overlap: int = 50) -> Iterator[Chunk]:
    for f in fiches:
        yield from chunk_fiche(f, size, overlap)

"""Turn DILA XML fiches into clean sections.

A fiche is a <Publication ID="F1234" spUrl="..."> with a <dc:title>, an <Introduction> and either a
<Texte> or, for "conditionnée" fiches, a <ListeSituations> of <Situation> (e.g. "En France" / "À
l'étranger"), each with its own <Texte>. A <Texte> is made of <Chapitre>/<SousChapitre>/<BlocCas>/<Cas>
(each with a <Titre>) containing <Paragraphe>, <Liste>/<Item> and <Tableau>. Navigation blocks (related
links, online services, references) are skipped.
The parser only relies on local tag names, so it tolerates namespaces and minor schema changes.
"""

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional
from xml.etree import ElementTree as ET

SECTION_TAGS = {"Situation", "Chapitre", "SousChapitre", "BlocCas", "Cas", "ASavoir", "Attention", "ANoter"}
SKIP_TAGS = {"FilDAriane", "Theme", "SousThemePere", "DossierPere", "SousDossierPere", "VoirAussi",
             "ServiceEnLigne", "PourEnSavoirPlus", "Reference", "OuSAdresser", "QuestionReponse",
             "Partenaire", "TitreRiche", "Condition", "LienExterneCommente", "QuiPeutMAider", "Actualite",
             "InformationComplementaire", "Abreviation", "Definition",
             "Montant", "RefActualite", "Audience", "Canal", "Avertissement", "LienWeb", "LienInterne"}
TEXT_TAGS = {"Paragraphe", "Item", "Ligne"}


def keep_type(t: str) -> bool:
    """Practical guides only ('Fiche d'information conditionnée', 'Fiche Question-réponse conditionnée'...).
    Guides that are only lists of external links carry no text worth indexing."""
    return t == "" or (t.startswith("Fiche") and "liens externes" not in t)


@dataclass
class Section:
    heading: List[str]
    text: str


@dataclass
class Fiche:
    id: str
    title: str
    url: str
    audience: str = ""
    sections: List[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _child(el: ET.Element, name: str) -> Optional[ET.Element]:
    for c in el:
        if local(c.tag) == name:
            return c
    return None


def _walk(el: ET.Element, heading: List[str], out: List[Section]) -> None:
    for c in el:
        tag = local(c.tag)
        if tag in SKIP_TAGS or tag == "Titre":
            continue
        if tag in SECTION_TAGS:
            t = _child(c, "Titre")
            _walk(c, heading + ([clean("".join(t.itertext()))] if t is not None else []), out)
        elif tag in TEXT_TAGS:
            txt = clean("".join(c.itertext()))
            if txt:
                txt = "- " + txt if tag == "Item" else txt
                if out and out[-1].heading == heading:
                    out[-1].text += "\n" + txt
                else:
                    out.append(Section(list(heading), txt))
        else:
            _walk(c, heading, out)


def parse_publication(root: ET.Element) -> Optional[Fiche]:
    if local(root.tag) != "Publication" or not keep_type(root.get("type", "")):
        return None
    title_el = next((e for e in root.iter() if local(e.tag) == "title"), None)
    title = clean("".join(title_el.itertext())) if title_el is not None else ""
    audience = clean("".join(_child(root, "Audience").itertext())) if _child(root, "Audience") is not None else ""
    fiche = Fiche(id=root.get("ID", ""), title=title, url=root.get("spUrl", ""), audience=audience)
    for part in ("Introduction", "Texte", "ListeSituations"):
        el = _child(root, part)
        if el is not None:
            _walk(el, [], fiche.sections)
    return fiche if fiche.sections else None


def parse_file(path: str) -> Iterator[Fiche]:
    """A file may hold one <Publication> or a wrapper containing many."""
    root = ET.parse(path).getroot()
    pubs = [root] if local(root.tag) == "Publication" else [e for e in root if local(e.tag) == "Publication"]
    for pub in pubs:
        fiche = parse_publication(pub)
        if fiche is not None:
            yield fiche


def iter_fiches(raw_dir: str) -> Iterator[Fiche]:
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.xml"))):
        try:
            yield from parse_file(path)
        except ET.ParseError as e:
            print(f"skipped {path}: {e}")

"""Tests without network, GPU or model downloads (pure-Python parts)."""

import io
import os
import tempfile
import zipfile
from xml.etree import ElementTree as ET

from eval.metrics import first_rank, summary
from rag_sp.chunk import chunk_fiche, split_words
from rag_sp.download import unpack
from rag_sp.generate import build_messages, cited_sources, normalize_citations, retry_delay
from rag_sp.parse import parse_file, parse_publication

FICHE = """<?xml version="1.0" encoding="UTF-8"?>
<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1234" type="Fiche d'information conditionnée"
             spUrl="https://www.service-public.fr/particuliers/vosdroits/F1234">
  <dc:title>Carte d'identité : renouvellement</dc:title>
  <FilDAriane><Niveau>Accueil particuliers</Niveau></FilDAriane>
  <Introduction><Texte><Paragraphe>La carte est valable <MiseEnEvidence>15 ans</MiseEnEvidence>.</Paragraphe></Texte></Introduction>
  <Texte>
    <Chapitre><Titre><Paragraphe>Où faire la demande ?</Paragraphe></Titre>
      <Paragraphe>En mairie, sur rendez-vous.</Paragraphe>
      <SousChapitre><Titre><Paragraphe>Pièces à fournir</Paragraphe></Titre>
        <Liste type="puce"><Item><Paragraphe>Une photo</Paragraphe></Item><Item><Paragraphe>Un justificatif de domicile</Paragraphe></Item></Liste>
      </SousChapitre>
    </Chapitre>
  </Texte>
  <ServiceEnLigne><Titre>Pré-demande</Titre></ServiceEnLigne>
</Publication>"""


def fiche():
    return parse_publication(ET.fromstring(FICHE))


# --- parsing ---------------------------------------------------------------
def test_parse_title_url_and_sections():
    f = fiche()
    assert (f.id, f.title) == ("F1234", "Carte d'identité : renouvellement")
    assert f.url.endswith("/F1234")
    assert [s.heading for s in f.sections] == [[], ["Où faire la demande ?"],
                                               ["Où faire la demande ?", "Pièces à fournir"]]
    assert f.sections[0].text == "La carte est valable 15 ans."          # inline tags kept as text
    assert f.sections[2].text == "- Une photo\n- Un justificatif de domicile"


def test_situations_become_headings():
    xml = FICHE.replace("<Texte>\n    <Chapitre>", "<ListeSituations><Situation><Titre>En France</Titre>"
                        "<TitreRiche><Paragraphe>En France</Paragraphe></TitreRiche><Texte>\n    <Chapitre>")
    xml = xml.replace("</Chapitre>\n  </Texte>", "</Chapitre>\n  </Texte></Situation></ListeSituations>")
    f = parse_publication(ET.fromstring(xml))
    assert f.sections[1].heading == ["En France", "Où faire la demande ?"]
    assert f.sections[2].heading[-1] == "Pièces à fournir"


def test_navigation_blocks_are_skipped():
    text = fiche().text
    assert "Accueil particuliers" not in text and "Pré-demande" not in text


def test_other_publication_types_are_ignored():
    root = ET.fromstring(FICHE.replace("Fiche d'information conditionnée", "Fiche avec liens externes"))
    assert parse_publication(root) is None


def test_parse_file_with_wrapper():
    body = FICHE.split("?>", 1)[1]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "all.xml")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(f"<Publications>{body}{body.replace('F1234', 'F9')}</Publications>")
        assert [f.id for f in parse_file(p)] == ["F1234", "F9"]


# --- chunking --------------------------------------------------------------
def test_split_words_overlap():
    words = " ".join(str(i) for i in range(10))
    assert split_words(words, 4, 1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert split_words("a b", 4, 1) == ["a b"]


def test_small_sections_are_packed_with_inline_subheadings():
    chunks = list(chunk_fiche(fiche(), size=300, overlap=50))
    assert len(chunks) == 1 and chunks[0].heading == ""
    assert chunks[0].text.split("\n") == [
        "La carte est valable 15 ans.", "[Où faire la demande ?]", "En mairie, sur rendez-vous.",
        "[Où faire la demande ? > Pièces à fournir]", "- Une photo", "- Un justificatif de domicile"]


def test_chunk_heading_is_the_shared_path():
    chunks = list(chunk_fiche(fiche(), size=12, overlap=2))
    assert [c.id for c in chunks] == ["F1234-0", "F1234-1"]
    assert chunks[0].heading == "" and "[Où faire la demande ?]" in chunks[0].text
    assert chunks[1].heading == "Où faire la demande ? > Pièces à fournir"
    assert chunks[1].for_embedding() == ("Carte d'identité : renouvellement > Où faire la demande ? > "
                                         "Pièces à fournir\n- Une photo\n- Un justificatif de domicile")


# --- generation helpers ----------------------------------------------------
PASSAGES = [{"fiche_id": "F1", "title": "A", "url": "u1", "heading": "h", "text": "t1"},
            {"fiche_id": "F1", "title": "A", "url": "u1", "heading": "", "text": "t2"},
            {"fiche_id": "F2", "title": "B", "url": "u2", "heading": "", "text": "t3"}]


def test_prompt_numbers_passages():
    user = build_messages("Q ?", PASSAGES)[1]["content"]
    assert "[1] A > h\nt1" in user and "[3] B\nt3" in user and user.endswith("Question : Q ?")


def test_cited_sources_dedupe_and_ignore_bad_numbers():
    assert cited_sources("x [3] y [1][2] z [9] [1]", PASSAGES) == [
        {"refs": [3], "title": "B", "url": "u2"}, {"refs": [1, 2], "title": "A", "url": "u1"}]


def test_native_citation_format_is_normalized():
    assert normalize_citations("120 €【1†L4-L7】【2】 et [3]") == "120 €[1][2] et [3]"


def test_retry_delay_from_message():
    assert retry_delay("Please try again in 12.15s.") == 12.65
    assert retry_delay("Please try again in 1m3.5s.") == 64.0
    assert retry_delay("nothing", default=7) == 7


# --- download --------------------------------------------------------------
def test_unpack_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("vosdroits/F1.xml", "<a/>")
        z.writestr("vosdroits/readme.txt", "x")
    with tempfile.TemporaryDirectory() as d:
        assert unpack(buf.getvalue(), d) == 1
        assert os.listdir(d) == ["F1.xml"]


# --- evaluation metrics ----------------------------------------------------
def test_metrics():
    assert first_rank("F2", ["F1", "F2", "F2"]) == 2 and first_rank("F9", ["F1"]) == 0
    assert summary([1, 2, 0, 4], ks=(1, 3)) == {"hit@1": 0.25, "hit@3": 0.5, "mrr": 0.438}


if __name__ == "__main__":  # runnable without pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

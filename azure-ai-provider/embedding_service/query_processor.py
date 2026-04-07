"""
query_processor.py — NLP Rule-based Query Parser (CPU-optimized, <50ms)
========================================================================

Replaces LLM parsing with spaCy + WordNet pipeline:
  1. Regex: extract OCR text from "quotes" or 'quotes'
  2. spaCy: POS tag → NOUN only + dependency parse → NUM counts
  3. WordNet: domain filter (physical objects) + Top-3 synonyms
  4. Build BM25 search strings for object + OCR vectors

Performance: <50ms on 4 vCPU (models pre-loaded, WordNet LRU-cached).
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# ── Lazy singletons ──────────────────────────────────────────────────────────
_nlp = None
_wn_ready = False

# Physical/visible object domains only
VALID_LEXNAMES = frozenset({
    # Core physical objects (original)
    "noun.artifact",   # man-made objects: table, bowl, shirt, car, sign
    "noun.person",     # person, officer, woman, man
    "noun.animal",     # cat, dog, bird, horse, elephant
    "noun.food",       # soup, rice, bread, fruit
    "noun.plant",      # tree, flower, grass
    "noun.body",       # hand, face, head, arm, leg
    "noun.object",     # generic physical objects
    "noun.substance",  # water, mud, wood, concrete
    # Additional YOLO-relevant categories
    "noun.group",      # people, crowd, herd, flock, team → YOLO counts groups
    "noun.shape",      # sphere, cube, cone → geometric objects in scenes
    "noun.location",   # street, road, sidewalk, area → YOLO scene context
})

STOP_NOUNS = frozenset({
    "scene", "background", "view", "way", "thing",
    "time", "part", "end", "moment", "shot", "frame",
    "setting", "exterior", "middle", "center",
})


OCR_PATTERN = re.compile(r"""["'\u201c\u201d\u2018\u2019]([^"'\u201c\u201d\u2018\u2019]+)["'\u201c\u201d\u2018\u2019]""")

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def initialize():
    """Pre-load spaCy + WordNet at startup."""
    global _nlp, _wn_ready
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        logger.info("spaCy en_core_web_sm loaded.")

    if not _wn_ready:
        import nltk
        for res in ["wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        from nltk.corpus import wordnet as wn
        wn.synsets("person")  # warm cache
        _wn_ready = True
        logger.info("WordNet loaded.")


@lru_cache(maxsize=2048)
def _is_valid(word: str) -> bool:
    from nltk.corpus import wordnet as wn
    for syn in wn.synsets(word, pos=wn.NOUN)[:3]:
        if syn.lexname() in VALID_LEXNAMES:
            return True
    return False


@lru_cache(maxsize=2048)
def _synonyms(word: str, k: int = 3) -> Tuple[str, ...]:
    from nltk.corpus import wordnet as wn
    out: Set[str] = set()
    for syn in wn.synsets(word, pos=wn.NOUN)[:3]:
        for lem in syn.lemmas():
            name = lem.name().replace("_", " ").lower()
            if name != word.lower() and len(name) > 2:
                out.add(name)
            if len(out) >= k:
                return tuple(sorted(out)[:k])
    return tuple(sorted(out)[:k])


@dataclass
class QueryAnalysis:
    original_query: str
    objects: List[dict]           # full objects with synonyms
    object_counts: dict           # {object_name: count} — only objects WITH a count
    ocr_texts: List[str]
    object_search_text: str
    ocr_search_text: str


def process_query(text: str) -> QueryAnalysis:
    """Parse query → objects + OCR extraction. Target: <50ms."""
    global _nlp, _wn_ready
    if _nlp is None or not _wn_ready:
        initialize()

    # 1. OCR from quotes
    ocr_texts = OCR_PATTERN.findall(text)
    clean = OCR_PATTERN.sub("", text).strip()

    # 2. spaCy
    doc = _nlp(clean)
    objects, seen = [], set()

    for tok in doc:
        if tok.pos_ != "NOUN":
            continue
        lem = tok.lemma_.lower()
        if lem in seen or lem in STOP_NOUNS or len(lem) < 3:
            continue
        if not _is_valid(lem):
            continue
        seen.add(lem)

        # Count from NUM child
        count = None
        for ch in tok.children:
            if ch.pos_ == "NUM" and ch.dep_ in ("nummod", "quantmod"):
                try:
                    count = int(ch.text)
                except ValueError:
                    count = _WORD_NUMS.get(ch.text.lower())
                break

        syns = list(_synonyms(lem))
        objects.append({"object": lem, "count": count, "synonyms": syns})

    # 3. Build BM25 strings
    terms = []
    for o in objects:
        terms.append(o["object"])
        terms.extend(o["synonyms"])
    obj_text = " ".join(terms) if terms else clean
    ocr_text = " ".join(ocr_texts)

    # 4. object_counts — {name: count} for objects with explicit numeric count
    object_counts = {
        o["object"]: o["count"]
        for o in objects
        if o["count"] is not None
    }

    return QueryAnalysis(
        original_query=text,
        objects=objects,
        object_counts=object_counts,
        ocr_texts=ocr_texts,
        object_search_text=obj_text,
        ocr_search_text=ocr_text,
    )

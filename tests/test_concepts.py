import json
from pathlib import Path

from ingestion.concepts import (
    BROAD_CATEGORIES,
    UNCATEGORIZED,
    concept_to_category,
    concepts_to_categories,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_exactly_15_categories():
    assert len(BROAD_CATEGORIES) == 15
    assert len(set(BROAD_CATEGORIES)) == 15


def test_all_raw_concepts_covered():
    raw = json.loads((REPO_ROOT / "data" / "raw" / "live_concepts_taxonomy_raw.json").read_text())
    all_concepts = [c for lst in raw.values() for c in lst]
    assert len(all_concepts) == 256
    uncategorized = [c for c in all_concepts if concept_to_category(c) == UNCATEGORIZED]
    assert uncategorized == []


def test_unknown_concept_falls_back_gracefully():
    assert concept_to_category("some brand new concept nobody has seen") == UNCATEGORIZED


def test_concepts_to_categories_dedupes_and_preserves_order():
    result = concepts_to_categories(["prime", "divisibility", "prime factorization"])
    # "prime" and "prime factorization" both -> Number Theory: Primes...
    # "divisibility" -> Number Theory: Divisibility...
    assert len(result) == 2
    assert result[0] != result[1]


def test_case_and_apostrophe_insensitive():
    a = concept_to_category("Vieta's Formulas")
    b = concept_to_category("vieta’s formulas")
    assert a == b != UNCATEGORIZED

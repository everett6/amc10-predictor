"""Sanitization helpers for text scraped from LIVE by Po-Shen Loh pages.

The site embeds problem statements/solutions as LaTeX-ish strings inside
JSON (Next.js `__NEXT_DATA__`). We keep the LaTeX (useful for rendering)
but strip anything that looks like it could be unsafe if ever rendered
as raw HTML, and normalize whitespace/unicode punctuation introduced by
the site's typography (curly quotes, non-breaking spaces, thin commas
in numbers like "17{,}402").
"""
from __future__ import annotations

import html
import re

_SCRIPT_TAG_RE = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_TAG_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
# Figures (and, for a few problems, the answer choices themselves) are
# <img> tags pointing at site-relative SVGs. Keep a text placeholder with
# the path so an image-only choice doesn't sanitize down to "".
_IMG_TAG_RE = re.compile(r"<img\b[^>]*?\bsrc=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)
_MULTI_WS_RE = re.compile(r"[ \t\f\v]+")
_LATEX_THIN_COMMA_RE = re.compile(r"\{,\}")

_UNICODE_REPLACEMENTS = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "--",
    " ": " ",
}


def strip_unsafe_markup(text: str) -> str:
    """Remove script/style tags and any remaining HTML tags."""
    if not text:
        return ""
    text = _SCRIPT_TAG_RE.sub("", text)
    text = _STYLE_TAG_RE.sub("", text)
    text = _IMG_TAG_RE.sub(lambda m: f"[image: {m.group(1)}]", text)
    text = _ANY_TAG_RE.sub("", text)
    return text


def normalize_text(text: str) -> str:
    """Normalize whitespace and typographic unicode; unescape HTML entities."""
    if text is None:
        return ""
    text = html.unescape(text)
    for bad, good in _UNICODE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = _MULTI_WS_RE.sub(" ", text)
    return text.strip()


def clean_latex_number(text: str) -> str:
    """Turn LaTeX '17{,}402' style grouping into plain '17,402'."""
    return _LATEX_THIN_COMMA_RE.sub(",", text)


def sanitize_problem_field(text: str) -> str:
    """Full pipeline applied to any scraped problem/solution/answer field."""
    text = strip_unsafe_markup(text)
    text = normalize_text(text)
    text = clean_latex_number(text)
    return text


def sanitize_concepts(raw_concepts: str) -> list[str]:
    """LIVE stores concepts as a single '; '-joined string; split + clean."""
    if not raw_concepts:
        return []
    parts = [normalize_text(p) for p in raw_concepts.split(";")]
    return [p for p in parts if p]

"""
pipeline/pdf_extractor.py — Step 1: PDF → list of clause strings.

WHAT IT DOES:
    1. Opens the uploaded PDF with pdfplumber.
    2. Extracts raw text page by page.
    3. Splits the raw text into individual clauses using a heuristic
       sentence / numbered-paragraph splitter.
    4. Cleans and filters trivially short fragments.

INTERVIEW TALKING POINTS:

WHY pdfplumber over PyPDF2 / pypdf?
    pdfplumber is built on pdfminer.six and preserves spatial layout
    information (character bounding boxes). This means it handles
    multi-column PDFs, tables, and indented sub-clauses more faithfully
    than pypdf, which is optimised for page-level operations (merge/split)
    rather than granular text extraction. For a contract analyser,
    layout fidelity directly affects whether a "clause" is extracted as
    one coherent unit or chopped mid-sentence.

LIMITATIONS:
    • Scanned / image-only PDFs produce no text — the user must OCR first.
    • Clause splitting is heuristic: it looks for numbered paragraphs
      (1., 2.1, (a), Article III) and sentence boundaries. Edge cases
      like run-on clauses spanning multiple numbered items will be split
      imperfectly. A production system would use a fine-tuned clause
      segmentation model (e.g. trained on CUAD).
    • pdfplumber can struggle with PDF files that use custom encoding or
      heavily ligature-heavy fonts — rare in NDAs but worth noting.
"""

from __future__ import annotations

import re
import io
from typing import List

import pdfplumber


# ── Constants ──────────────────────────────────────────────────────────────────

# Patterns that suggest the start of a new clause.
# The regex covers:
#   • Numbered paragraphs:  1.  /  2.3  /  10.
#   • Sub-items:            (a)  /  (i)
#   • "Article" / "Section" headings
#   • ALL-CAPS headings (common in NDAs)
_CLAUSE_BREAK_RE = re.compile(
    r"""
    (?:^|\n)                          # start of string or newline
    (?:
        \d+(?:\.\d+)*\.?\s            # 1. / 1.2 / 1.2.3
      | \([a-zA-Z]{1,3}\)\s           # (a) / (iv)
      | (?:Article|Section|Clause)\s+\S+\s  # Article IV / Section 3.2
      | [A-Z][A-Z\s]{4,}(?:\n|:)     # ALL CAPS HEADING
    )
    """,
    re.VERBOSE | re.MULTILINE,
)

MIN_CLAUSE_WORDS = 8   # discard fragments shorter than this
MAX_CLAUSE_CHARS = 2000  # very long blocks are likely merged paragraphs; split at sentence level too


# ── Public API ──────────────────────────────────────────────────────────────────

def extract_clauses(uploaded_file) -> List[str]:
    """
    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile (file-like object)

    Returns
    -------
    List[str]
        Each element is one clause or paragraph-level text block.
        Order is preserved (top-to-bottom, page-by-page).
    """
    raw_text = _extract_raw_text(uploaded_file)
    clauses = _split_into_clauses(raw_text)
    return clauses


# ── Internal helpers ────────────────────────────────────────────────────────────

def _extract_raw_text(uploaded_file) -> str:
    """Read all pages of a PDF and concatenate their text."""
    # Streamlit UploadedFile is a BytesIO-compatible object.
    # We wrap it in io.BytesIO to guarantee seek() support.
    pdf_bytes = io.BytesIO(uploaded_file.read())

    pages_text: List[str] = []
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                pages_text.append(text.strip())

    # Join pages with double newline so page boundaries act as potential
    # clause-break markers.
    return "\n\n".join(pages_text)


def _split_into_clauses(text: str) -> List[str]:
    """
    Heuristic clause splitter.

    Strategy:
        1. Try to split on numbered-paragraph / heading patterns.
        2. For any resulting chunk that is still very long, further split
           on sentence boundaries as a fallback.
        3. Discard fragments that are too short to be meaningful clauses.
    """
    # Step 1: find all clause-break positions
    break_positions = [m.start() for m in _CLAUSE_BREAK_RE.finditer(text)]

    if len(break_positions) < 2:
        # No numbered structure found — fall back to sentence splitting
        return _sentence_split_fallback(text)

    # Build clause list from break positions
    raw_chunks: List[str] = []
    for i, start in enumerate(break_positions):
        end = break_positions[i + 1] if i + 1 < len(break_positions) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            raw_chunks.append(chunk)

    # Step 2: further split any chunk that is suspiciously long
    clauses: List[str] = []
    for chunk in raw_chunks:
        if len(chunk) > MAX_CLAUSE_CHARS:
            clauses.extend(_sentence_split_fallback(chunk))
        else:
            clauses.append(chunk)

    # Step 3: filter noise
    clauses = [_clean(c) for c in clauses]
    clauses = [c for c in clauses if len(c.split()) >= MIN_CLAUSE_WORDS]

    return clauses


def _sentence_split_fallback(text: str) -> List[str]:
    """
    Split text into sentences using a simple regex.
    Used when paragraph structure is absent or a chunk is too large.

    INTERVIEW NOTE: A production system would use spaCy's sentencizer or
    a pre-trained sentence boundary detection model for higher accuracy.
    We avoid that dependency here to keep the Hugging Face Spaces build
    lightweight and fast.
    """
    # Split on ". " / ".\n" followed by a capital letter
    sentence_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_re.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _clean(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\x20-\x7E\n]', '', text)  # strip non-ASCII
    return text.strip()

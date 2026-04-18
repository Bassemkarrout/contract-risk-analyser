---
title: NDA Contract Risk Analyser
emoji: ⚖️
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# ⚖️ NDA Contract Risk Analyser

A graduate-level AI pipeline that analyses NDA contracts for legal risk.

## What it does

1. **PDF Parsing** — Extracts and segments clauses from an uploaded NDA using `pdfplumber`.
2. **Risk Classification** — Uses a HuggingFace zero-shot classification model (`facebook/bart-large-mnli`) to assign each clause a risk category (liability, indemnification, termination, penalty, exclusivity, confidentiality) with a confidence score.
3. **Risk Aggregation** — Computes a weighted overall risk score (0–100) and bands it as Low / Medium / High.
4. **Plain-English Explanations** — Sends flagged clauses to the Gemini API via LangChain and returns structured explanations of *why* each clause is risky and *what to negotiate*.
5. **Interactive Dashboard** — Displays everything in a colour-coded Streamlit UI.

## Tech stack

| Layer | Technology | Reason |
|---|---|---|
| PDF parsing | `pdfplumber` | Superior layout-preserving text extraction vs. pypdf |
| Clause classification | `facebook/bart-large-mnli` (HuggingFace zero-shot) | No labelled training data needed; interpretable confidence scores |
| LLM explanations | Gemini 1.5 Flash via `langchain-google-genai` | Fast, generous free tier, LangChain abstraction for vendor flexibility |
| UI | Streamlit | Rapid prototype-to-demo for data-heavy Python apps |
| Deployment | Hugging Face Spaces | Native transformers model caching, free hosting |

## Setup

### Local

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/nda-risk-analyser
cd nda-risk-analyser
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
streamlit run app.py
```

### Hugging Face Spaces

1. Fork this Space.
2. Add your `GOOGLE_API_KEY` in **Settings → Repository secrets**.
3. The Space will build automatically. First run downloads the `~400 MB` BART model — subsequent runs use the cache.

## Architecture

```
app.py  (Streamlit entry point)
│
├── pipeline/
│   ├── pdf_extractor.py    # pdfplumber → list of clause strings
│   ├── classifier.py       # HF zero-shot → ClassifiedClause objects
│   ├── risk_aggregator.py  # weighted scoring → RiskSummary
│   └── explainer.py        # LangChain + Gemini → plain-English JSON
│
└── ui/
    └── dashboard.py        # Streamlit rendering (separated for testability)
```

## Limitations & future work

- **Scanned PDFs**: pdfplumber cannot OCR image-only PDFs. A production version would add Tesseract OCR as a fallback.
- **Classification accuracy**: Zero-shot classification is a pragmatic choice for zero labelled data. Fine-tuning on [CUAD](https://www.atticusprojectai.org/cuad) (13k labelled contract clauses) would substantially improve F1.
- **LLM hallucination**: Gemini explanations are AI-generated and not a substitute for qualified legal advice.
- **Multi-lingual NDAs**: The BART model is English-only. `joeddav/xlm-roberta-large-xnli` supports 100+ languages.
- **Clause segmentation**: The heuristic regex splitter handles standard numbered NDAs. Complex formatting (tables, multi-column) may require a trained segmentation model.

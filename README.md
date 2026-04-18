# NDA Contract Risk Analyser

A portfolio project I built to demonstrate an end-to-end AI pipeline for legal document analysis. The app takes an NDA PDF, breaks it down clause by clause, flags the risky ones, and explains why they're risky in plain English — no legal background needed to understand the output.

---

## What it does

1. You upload an NDA PDF through the web interface
2. The app extracts and segments all clauses from the document
3. Each clause gets classified into a risk category with a confidence score using a HuggingFace zero-shot model
4. The scores are aggregated into an overall risk rating (Low / Medium / High) using worst-clause weighted scoring — so one genuinely dangerous clause always surfaces, it doesn't get diluted by surrounding standard clauses
5. Flagged clauses are sent to Llama 3.3 via Groq which generates a plain-English explanation of the risk and what to negotiate
6. Everything is displayed in a colour-coded Streamlit dashboard

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| PDF parsing | pdfplumber | Better layout preservation than pypdf for legal documents |
| Clause classification | facebook/bart-large-mnli | Zero-shot classification, no labelled training data needed |
| LLM explanations | Llama 3.3 70B via Groq + LangChain | Fast, generous free tier, LangChain abstracts the provider |
| UI | Streamlit | Fastest way to ship a clean data-heavy Python interface |
| Deployment | Hugging Face Spaces | Native model caching, free hosting |

---

## Risk categories

The classifier assigns each clause to one of six categories:

- **Liability** — clauses that limit or cap what one party owes the other
- **Indemnification** — clauses that make one party responsible for the other's legal costs
- **Termination** — conditions under which the agreement can be ended
- **Penalty** — liquidated damages or financial penalties for breach
- **Exclusivity** — restrictions on working with third parties or competitors
- **Confidentiality** — obligations around keeping information secret

Liability and indemnification are weighted highest in the overall score because they carry the most financial exposure in a typical NDA.

---

## Run locally

```bash
git clone https://github.com/Bassemkarrout/contract-risk-analyser
cd contract-risk-analyser
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
python -m streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card needed.

---

## Known limitations

- Scanned or image-only PDFs won't work — the document needs a text layer for pdfplumber to extract anything
- Zero-shot classification gets around 60-70% F1 on legal text. The production upgrade would be fine-tuning on CUAD (13,000 labelled contract clauses) which pushes accuracy to 85-90%
- LLM explanations are AI-generated and not a substitute for actual legal advice
- The BART model is English-only

---

## Project structure

```
contract_risk_analyser/
├── app.py                        
├── pipeline/
│   ├── pdf_extractor.py          
│   ├── classifier.py             
│   ├── risk_aggregator.py        
│   └── explainer.py              
└── ui/
    └── dashboard.py              
```

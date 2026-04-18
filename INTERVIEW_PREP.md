# PwC Graduate Interview Prep: NDA Risk Analyser

This document maps every likely interview question to a defensible answer
grounded in the actual technical choices made in this project.

---

## 1. "Walk me through the project end-to-end."

Use the pipeline metaphor — interviewers love a clear mental model:

> "The pipeline has four stages. First, a user uploads an NDA PDF through a
> Streamlit web app. Second, pdfplumber extracts the text and a regex-based
> splitter segments it into individual clauses. Third, a HuggingFace
> zero-shot classification model categorises each clause as liability,
> indemnification, termination, penalty, exclusivity, or confidentiality,
> and gives a confidence score. Fourth, clauses above a confidence threshold
> are sent to Google's Gemini model via LangChain, which returns a structured
> JSON explanation of why each clause is risky and what to negotiate. The
> results are displayed in a Streamlit dashboard with a colour-coded risk
> score at the top."

---

## 2. "Why did you choose pdfplumber over PyPDF2?"

> "Both can extract text, but pdfplumber is built on pdfminer.six and
> preserves spatial layout — it knows where characters are on the page in
> two-dimensional space. This matters for contracts because legal documents
> often have indented sub-clauses, side-by-side columns, and numbered
> hierarchies. PyPDF2 is better optimised for page-level operations like
> merge and split. For text extraction fidelity, pdfplumber wins. The main
> limitation is that neither handles scanned PDFs — you'd need Tesseract
> OCR as a pre-processing step for those."

---

## 3. "Why zero-shot classification? Why not a supervised model?"

> "Zero-shot classification was the pragmatic choice because I had no
> labelled training data — I couldn't afford to annotate hundreds of NDA
> clauses as a student. The model I used, facebook/bart-large-mnli, was
> trained on the Multi-Genre NLI dataset and learns to ask 'does this text
> entail this description?' This lets me define risk categories as English
> sentences and classify without any fine-tuning.
>
> The production upgrade would be fine-tuning on CUAD — the Contract
> Understanding Atticus Dataset, which has 13,000 expert-labelled clauses
> across 41 legal question types. That would likely take F1 from around
> 60–70% (typical for ZSC on legal text) to 85–90%."

---

## 4. "What are the limitations of your confidence scores?"

> "The scores are calibrated to the NLI task, not directly to 'probability
> of being legally risky'. A confidence of 0.72 means the NLI model is 72%
> confident the text entails the category description — it's a relative
> signal, not a calibrated probability. Additionally, the model can confuse
> structurally similar clauses. A confidentiality clause that also mentions
> financial penalties might score high on both categories. In production
> you'd want Platt scaling or isotonic regression to calibrate the scores,
> and human expert validation to set threshold values."

---

## 5. "Why LangChain? Couldn't you call Gemini directly?"

> "Yes, I could have used the google-generativeai SDK directly in two lines.
> I chose LangChain for three reasons that are especially relevant in a
> consulting context.
>
> First, vendor abstraction: LangChain wraps multiple LLM providers behind
> the same interface. If the client wants to switch from Gemini to GPT-4 or
> Claude, you change one line — not the entire pipeline. PwC advises clients
> across different cloud providers; avoiding vendor lock-in is a genuine
> concern.
>
> Second, prompt management: LangChain's PromptTemplate separates prompt
> logic from application logic. In a team, this means a lawyer or product
> manager can iterate on prompts without touching Python code.
>
> Third, extensibility: the same chain architecture supports adding RAG
> (retrieval-augmented generation), memory, or multi-step reasoning later.
> For this MVP it's a single-turn chain, but the architecture is ready to
> scale."

---

## 6. "How does your risk scoring work? Is it statistically valid?"

> "The scoring is a weighted average. Each clause contributes
> confidence × category_weight to its category bucket. I then normalise
> across all clauses and map to 0–100. The weights reflect business logic:
> liability and indemnification get 25% each because they directly determine
> financial exposure; confidentiality gets 10% because it's expected in
> every NDA and doesn't signal unusual risk.
>
> It's not statistically validated — the weights are hand-crafted, not
> learned from data. The thresholds (Low <40, Medium 40–70, High ≥70) are
> arbitrary without ground truth. A production system would train a small
> logistic regression on lawyer-labelled NDAs to learn both the weights and
> the thresholds. This is a known limitation I'd highlight to a client."

---

## 7. "What could go wrong with the LLM explanations?"

> "Three main risks. First, hallucination — Gemini might generate plausible-
> sounding but legally incorrect explanations. In a production legal tool,
> every AI-generated explanation would be reviewed by a qualified lawyer
> before surfacing to end users, and there'd be a disclaimer that this is
> not legal advice. Second, non-determinism — even at temperature 0.2, the
> same clause can produce slightly different explanations across runs. For
> auditability you'd want to log and version all LLM outputs. Third, prompt
> injection — a malicious contract could embed instructions in a clause
> designed to manipulate the LLM's output. Input sanitisation and output
> validation against the expected JSON schema are mitigations."

---

## 8. "Why Streamlit? What are its limitations?"

> "Streamlit was chosen for speed-to-demo. It lets you build a polished
> data-heavy UI in pure Python with no front-end code, which is ideal for a
> prototype. The limitations are real: it's single-threaded, so with a large
> PDF and 50 clauses, the HuggingFace model runs synchronously and the user
> sees a spinner. A production version would use FastAPI as the backend with
> async task queuing (Celery or Cloud Tasks), and a proper React front end.
> Streamlit also has limited state management for complex multi-user
> scenarios. For a CV demo targeting a non-technical interviewer at PwC,
> the trade-off was absolutely worth it."

---

## 9. "How would you deploy this at PwC scale?"

> "The current deployment on Hugging Face Spaces is a single-container
> proof of concept. For enterprise scale I'd redesign as:
>
> - **Containerisation**: Docker image with the BART model baked in,
>   deployed on GCP Cloud Run or Azure Container Apps for auto-scaling.
> - **Async processing**: Replace the synchronous pipeline with a message
>   queue (Pub/Sub or Azure Service Bus). User uploads trigger a job;
>   results are pushed back via WebSocket.
> - **Model serving**: Move the HuggingFace model to a dedicated inference
>   endpoint (Vertex AI or HF Inference Endpoints) to decouple model
>   latency from the web server.
> - **Security**: NDAs are confidential documents. All processing in a
>   client's private VPC, no data leaving the perimeter, audit logs for
>   every document processed.
> - **Accuracy**: Fine-tune on CUAD, implement human-in-the-loop review for
>   high-risk flags, track drift over time with MLflow or Weights & Biases."

---

## 10. "What would you do differently if you had more time?"

> "Three things. First, I'd fine-tune a smaller model — deberta-v3-base —
> on CUAD rather than using zero-shot. Smaller and more accurate. Second,
> I'd add a clause comparison feature: embed all clauses with sentence-
> transformers and use cosine similarity to find analogous clauses in a
> reference corpus of 'market standard' NDAs, so the tool tells you not
> just that a clause is risky but how unusual it is relative to industry
> norms. Third, I'd add proper evaluation: a test set of NDAs labelled by
> a lawyer, precision/recall metrics per category, and a confusion matrix
> so I can actually quantify how good the classifier is rather than relying
> on qualitative impressions."

---

## Key numbers to remember

| Fact | Value |
|---|---|
| BART-large-mnli model size | ~400 MB |
| Typical NDA clause count | 30–60 |
| Pipeline runtime (CPU) | ~45–90 seconds |
| CUAD dataset size | 13,000 labelled clauses, 510 contracts |
| ZSC typical F1 on legal text | ~60–70% |
| Fine-tuned CUAD F1 (state of art) | ~85–90% |
| Gemini 1.5 Flash context window | 1M tokens |
| Gemini free tier RPM | ~60 requests/min |

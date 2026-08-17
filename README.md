# enterprise-rag-assistant

## Persian Document Intelligence System

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed and built for **Pakshoo Industrial Group**, enabling employees to query internal documents, meeting minutes, and official memoranda in natural Persian language and receive accurate, context-grounded answers — without exposing sensitive organizational data to third-party APIs.

> **Note:** This repository contains the core application logic developed for this project. Data files, employee records, and document/vector indexes are intentionally excluded (see `.gitignore`) to preserve the confidentiality of real organizational data — only the application source is published. Configuration such as file paths and secrets is loaded from environment variables and is not hardcoded in source.

## Setup

1. **Clone the repo and set up a virtual environment**
   ```
   git clone https://github.com/MehrshadHaghighat007/enterprise-rag-assistant.git
   cd enterprise-rag-assistant
   python3 -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Create your `.env` file from the template and fill in real values**
   ```
   cp .env.example .env
   ```
   `.env.example` is a placeholder template safe to commit; `.env` holds your real secret key and local paths and must never be committed (it's already covered by `.gitignore`). At minimum, set:
   - `SECRET_KEY` — any long random string, used to sign Flask session cookies
   - `UPLOAD_FOLDER`, `USER_CHAT_DIR`, `EMPLOYEES_FILE` — local paths for this machine (relative paths like `./data/...` work fine and are created automatically)
   - `LLM_ENDPOINT` / `LLM_MODEL_NAME` — see local LLM setup below
   - `RAW_INDEX_FILE`, `RAW_REFS_FILE` — just pick any writable file paths; you don't create these files yourself, the app generates them automatically on first run (see below)

4. **Pre-download the embedding model**
   The app runs with `HF_HUB_OFFLINE=1` (no network calls at query time), so the multilingual sentence-transformer model needs to be cached locally *before* the first run:
   ```
   HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
   ```
   This downloads the model once into the Hugging Face cache (`~/.cache/huggingface`). After that, the app can run fully offline.

5. **Set up the local LLM (Ollama)**
   Install [Ollama](https://ollama.com), then pull the model referenced by `LLM_MODEL_NAME` in your `.env` (e.g. `ollama pull llama3:8b-instruct-q4_0`, or the equivalent Qwen tag for the production setup). Ollama serves on `http://localhost:11434` by default, matching `LLM_ENDPOINT` — make sure it's running before starting the app.

6. **Add an `employees.json`**
   Create the file at the path set in `EMPLOYEES_FILE`, mapping employee IDs to name/unit/role, e.g.:
   ```json
   {
     "1001": {"name": "Test User", "unit": "IT", "role": "employee"},
     "1002": {"name": "Admin", "unit": "IT", "role": "head_of_unit"}
   }
   ```

7. **Add PDF documents**
   Drop a few PDFs into `UPLOAD_FOLDER` so the RAG pipeline has content to index. Documents matching the expected Persian meeting-minutes headings (`تاریخ`, `موضوع جلسه`, `رئیس جلسه`, etc.) get full structured metadata extraction; other PDFs are still indexed for raw-text semantic search.

8. **Run the app**
   ```
   python web_app.py
   ```
   On first launch the app builds the metadata index and the raw-text FAISS index (`RAW_INDEX_FILE` / `RAW_REFS_FILE`) from your PDFs and saves them to the paths you chose in `.env` — you don't create these two files yourself. On every later run, they're loaded from disk instead of being rebuilt; delete them together if you ever want to force a full rebuild (e.g. after adding new PDFs). The app then starts the Flask dev server at `http://127.0.0.1:5000`.

## Overview

Organizations accumulate large volumes of unstructured Persian-language documents — meeting minutes, internal memos, compliance notices — that are difficult to search using traditional keyword tools, especially given the linguistic variability of Persian (multiple valid spellings, informal headings, inconsistent formatting). This system solves that problem end-to-end: it ingests raw PDF archives, understands their structure semantically, and lets employees ask natural-language questions and receive precise, sourced answers through a secure web interface.

The system was fully designed, implemented, and deployed by me, covering the entire stack: document processing, retrieval architecture, LLM integration, backend API, authentication, and frontend.

## Key Features

- **Dual-index semantic retrieval** — combines a structured *metadata index* (meeting subject, attendees, date, resolutions, etc.) with a *raw-text chunk index* over full document contents, giving the system both precise field-level lookup and broader semantic search over document bodies.
- **Persian-native NLP pipeline** — a dedicated text normalization layer resolves the many orthographic variants of Persian script (e.g. ی/ي, ک/ك, Arabic vs. Persian numerals) before indexing or matching, which is essential for reliable retrieval in Persian corpora.
- **Hybrid field-matching strategy** — user queries are resolved against document metadata using a three-stage cascade: exact keyword matching, fuzzy string matching (via sequence similarity) for typos and variant phrasing, and embedding-based semantic matching for queries that use entirely different wording than the source documents.
- **Local, privacy-preserving inference** — all language model inference runs on local infrastructure rather than external APIs, ensuring sensitive internal documents never leave the organization's environment. The production deployment is served by a locally-hosted **Qwen** LLM (an early development iteration used a locally-hosted LLaMA 3 model via Ollama).
- **Vector search at scale** — document embeddings are generated with `sentence-transformers` (multilingual MiniLM) and indexed with **FAISS** for fast approximate nearest-neighbor retrieval, with persistent on-disk index storage to avoid re-embedding on every restart.
- **PDF extraction pipeline** — automated text and metadata extraction from PDF meeting records using PyMuPDF, with regex-based structured field parsing tuned to the organization's document templates.
- **Conversation memory** — a SQLite-backed conversation store maintains multi-turn chat history per user, allowing employees to revisit past queries and threaded conversations, with a migration path implemented for evolving schema versions.
- **Role-based access control (RBAC)** — two distinct dashboards enforced at the route level:
  - **Head of Unit (admin) dashboard** — document upload/removal, operation logs, and feedback review.
  - **Employee dashboard** — natural-language search and conversational Q&A over the document base.
- **Full-stack ownership** — backend built with **Flask**, frontend built with **Bootstrap 5**, including AJAX-driven chat interactions for a responsive, non-reloading query experience.

## Architecture

```
PDF Documents
     │
     ▼
Extraction Layer (PyMuPDF) ── Persian text normalization
     │
     ├──► Metadata Index (per-field embeddings, FAISS)
     └──► Raw-Text Chunk Index (chunked full-text embeddings, FAISS)
     │
     ▼
Query Resolution (exact → fuzzy → semantic field matching)
     │
     ▼
Context Assembly ── Prompt Construction
     │
     ▼
Local LLM Inference (Qwen, self-hosted)
     │
     ▼
Flask Backend (RBAC, session mgmt, conversation storage)
     │
     ▼
Bootstrap 5 Frontend (role-based dashboards)
```

## Tech Stack

| Layer | Technologies |
|---|---|
| Language | Python |
| NLP / Embeddings | `sentence-transformers` (multilingual MiniLM) |
| Vector Search | FAISS |
| PDF Processing | PyMuPDF (fitz) |
| LLM Inference | Locally-hosted Qwen (production); LLaMA 3 via Ollama (early development) |
| Backend | Flask |
| Database | SQLite |
| Frontend | HTML, Bootstrap 5, JavaScript (AJAX) |
| Auth & Access Control | Session-based auth with role-based dashboard routing |

## My Role

I independently designed and implemented this system end-to-end for Pakshoo Industrial Group, including:
- Requirements analysis and system architecture design
- The full document ingestion and dual-index retrieval pipeline
- The Persian NLP normalization and hybrid query-matching logic
- Integration of local LLM inference to meet the organization's data-privacy constraints
- The complete backend (Flask) and frontend (Bootstrap 5) implementation
- Database schema design and the authentication / RBAC system

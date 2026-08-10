# Enterprise RAG Assistant — Persian Document Intelligence System

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed and built for **Pakshoo Industrial Group**, enabling employees to query internal documents, meeting minutes, and official memoranda in natural Persian language and receive accurate, context-grounded answers — without exposing sensitive organizational data to third-party APIs.

> **Note:** This repository contains the core application logic developed for this project. The data files, employee records, and document indexes included here (`employees.json`, `*.db`, `*.index`, `*.pkl`) are synthetic placeholders created to demonstrate the system's structure while preserving the confidentiality of real organizational data. Absolute file paths in the source reflect the original local development environment and are not intended to run as-is.

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

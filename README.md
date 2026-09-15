# Uni-Regulations Agentic Assistant

Bilingual (Vietnamese / English) RAG assistant over official university regulation PDFs.
Every answer carries its sources — document, section, and page — plus a confidence band, so a
student can verify the rule or know to ask an advisor instead.

- **Retrieval**: hybrid dense + BM25 over Zilliz Cloud (Milvus), fused with reciprocal rank fusion.
- **Generation**: Gemini, returning structured JSON so the answer and its own grounding rating
  arrive in a single round trip.
- **Bilingual**: the query language is detected; English questions are also embedded in
  Vietnamese translation so they retrieve from Vietnamese source documents.
- **Freshness**: weekly crawl with SHA-256 content-hash change detection; unchanged documents
  cost nothing to re-check.

## Architecture

```
presentation/  ──▶  application/  ──▶  domain/ports.py   (Protocols only)
(FastAPI, React)    (RAG, ingestion,          ▲
                     admin services)          │
infrastructure/ ─────────────────────────────-┘
(Gemini, Milvus, Selenium, PyMuPDF, SQLite)
```

`application/` imports only from `domain/`. Adapters implement the Protocols structurally,
without inheriting from them. `main.py` is the one module that knows both sides — swapping the
vector database or the model provider is a change to `build_container` and nothing else.

| Path | Holds |
|---|---|
| `config/settings.py` | All configuration. No other module reads the environment. |
| `domain/` | Models, exception hierarchy, and the port Protocols. |
| `application/` | `rag_service`, `ingestion_service`, `admin_service` — pure orchestration. |
| `infrastructure/` | The only modules importing third-party libraries. |
| `presentation/api/` | FastAPI routes, DI container, admin guard. |
| `frontend/` | React + Vite SPA (chat, source panel, confidence badge, admin dashboard). |

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # then fill in the required values
```

Required: `GEMINI_API_KEY`, `ZILLIZ_CLOUD_ENDPOINT`, `ZILLIZ_CLOUD_API_KEY`.
Set `ADMIN_TOKEN` to enable the admin API — left empty, those routes return 503 rather than
running unauthenticated.

The crawler drives headless Chrome through `webdriver-manager`, so Chrome must be installed.

```bash
cd frontend && npm install && npm run build   # writes frontend/dist, which FastAPI serves
```

## Running

```bash
.venv/bin/python main.py serve              # API + built frontend on :8000
.venv/bin/python main.py ingest --limit 1    # crawl and index one document
.venv/bin/python main.py ingest --all        # re-index everything, ignoring hashes
.venv/bin/python main.py status              # what is indexed, per domain
```

For frontend development, run `main.py serve` and `npm run dev` together — Vite proxies `/api`
to port 8000.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Ask a question. Body: `message`, optional `session_id`, optional `domain`. |
| `DELETE` | `/api/sessions/{id}` | Clear a conversation's history. |
| `GET` | `/api/domains` | The five regulatory domains, labelled in both languages. |
| `GET` | `/api/health` | Liveness. |
| `GET` | `/api/admin/status` | Index counts, per-domain totals, recent runs. |
| `GET` | `/api/admin/documents` | Every known document and its state. |
| `POST` | `/api/admin/crawl` | Trigger a crawl now. |
| `POST` | `/api/admin/documents` | Upload and index a PDF immediately. |

Admin routes require an `X-Admin-Token` header matching `ADMIN_TOKEN`.

```bash
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message":"Một học kỳ được đăng ký tối đa bao nhiêu tín chỉ?"}'
```

## Tests

```bash
.venv/bin/python -m pytest          # fully offline; every port is mocked
```

The suite covers the RAG pipeline (orchestration, confidence banding, language routing, session
context, failure mapping), ingestion (change detection, re-index hygiene, failure isolation, run
auditing), the chunker, the extraction quality gate, rank fusion, the registry, sessions, and both
HTTP surfaces including the auth boundary.

## Data and state

- **Vector index**: collection `uni_regulations_v1` on Zilliz Cloud. Dense (COSINE, 3072-dim) plus
  a BM25 function field, with `domain`, `section`, `page_start`/`page_end` and `doc_updated_at`
  stored for filtering and citation.
- **Registry**: `database/regulations.db` (SQLite) holds document state and ingestion run history.
  It replaces the legacy `seen_laws.json`, `processor_seen_laws.json`, and
  `processed_records.jsonl`.
- **Sessions**: in-process, with a TTL. Nothing about a conversation is persisted, and the UI
  exposes a clear-history control.

Ingestion is idempotent: a document whose hash is unchanged is skipped, a changed one has its old
chunks deleted before the new ones are written, and one failing document never aborts a run.

## Calibration

Two things are heuristics with real knobs, not settled numbers:

- **Confidence**: `CONFIDENCE_DENSE_WEIGHT` blends retrieval similarity with the model's grounding
  self-rating; `CONFIDENCE_HIGH_THRESHOLD` / `CONFIDENCE_MEDIUM_THRESHOLD` set the bands. These
  need tuning against an accuracy benchmark before launch.
- **Extraction gate**: `PYMUPDF_QUALITY_THRESHOLD` and `EXPECTED_CHARS_PER_PAGE` decide when a PDF
  is re-read by the model. Scoring measures text density per page and legibility, so a short but
  cleanly extracted document is accepted while a scanned or garbled one is not.

If a deployment's Milvus build has no usable analyzer for Vietnamese, set `ENABLE_SPARSE=false` to
index and search dense-only.

## License

MIT — see `LICENSE`.

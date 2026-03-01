# PDF Multimodal RAG

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline for complex PDF documents. Upload any PDF — including those with multi-column layouts, tables, and figures — and ask questions about its content in natural language. Responses are streamed in real-time with inline source citations.

Live at: **https://mm-pdf-rag-fh.duckdns.org**

---

## Features

- **Multimodal understanding** — tables extracted as structured data, images described via Claude Vision
- **Layout-aware parsing** — handles multi-column PDFs, nested headings, and mixed content using Docling
- **Hierarchical chunking** — chunks respect document structure, preserving section hierarchy as context
- **Semantic search** — Amazon Titan Embeddings v2 for high-quality vector retrieval
- **Streaming responses** — real-time token streaming via Server-Sent Events (SSE)
- **Source citations** — every response includes retrievable source cards (filename, section, page numbers, relevance score)
- **Session isolation** — each browser session manages its own document scope independently
- **Deduplication** — SHA-256 hashing prevents re-indexing identical documents
- **Background indexing** — upload returns immediately; progress is polled via a job status API

---

## Architecture

```
Browser
  │
  ▼
Nginx (port 80)
  ├── /          → React static build (web/dist/)
  └── /api/      → FastAPI (port 8000, internal)
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Docling    ChromaDB    AWS Bedrock
      (parsing)  (vector DB)  ├── Titan Embeddings
                              └── Claude 3.5 Sonnet
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| PDF Parsing | Docling |
| Chunking | Custom hierarchical strategy |
| Vector DB | ChromaDB (embedded) |
| Embeddings | AWS Bedrock — Amazon Titan Embed v2 |
| LLM | AWS Bedrock — Claude 3.5 Sonnet |
| Backend | FastAPI + Uvicorn |
| Frontend | React + TypeScript + Tailwind CSS v3 |
| Reverse Proxy | Nginx |
| Hosting | AWS EC2 (t3.large, Ubuntu 22.04) |

---

## Project Structure

```
PDF_Multimodal_RAG/
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app + lifespan
│   │   ├── models.py             # Pydantic request/response models
│   │   ├── registry.py           # Document registry (JSON)
│   │   └── routes/
│   │       ├── ingest.py         # Upload + background indexing
│   │       ├── documents.py      # List + delete documents
│   │       └── query.py          # Query + SSE streaming
│   ├── ingestion/
│   │   ├── pdf_parser.py         # Docling-based PDF parser
│   │   ├── indexing_pipeline.py  # Orchestrates parse → chunk → embed → store
│   │   └── chunking/
│   │       ├── hierarchical.py   # Section-aware chunker
│   │       └── semantic.py       # Sentence-embedding chunker
│   ├── vectordb/
│   │   └── chroma_client.py      # ChromaDB operations
│   ├── embeddings/
│   │   └── bedrock_embeddings.py # Titan Embed v2 via Bedrock
│   ├── retrieval/
│   │   └── retriever.py          # Embedding + similarity search
│   └── generation/
│       ├── rag_pipeline.py       # RAG orchestration + streaming
│       ├── response_generator.py # Claude via Bedrock
│       └── image_describer.py    # Claude Vision for PDF images
├── web/                          # React frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts                # API calls + SSE streaming
│   │   ├── types.ts
│   │   └── components/
│   │       ├── Sidebar.tsx       # Upload, document list, scope selector
│   │       └── ChatArea.tsx      # Chat interface with streaming
│   └── tailwind.config.js
├── deploy/
│   ├── nginx.conf                # Nginx site config
│   ├── pdf-rag.service           # systemd service
│   └── setup_ec2.sh              # EC2 provisioning script
├── frontend/
│   └── app.py                    # Streamlit UI (reference, not primary)
├── .env.example
└── requirements.txt
```

---

## Local Development

### Prerequisites
- Python 3.12
- Node.js 20+
- AWS account with Bedrock access (Titan Embed v2 + Claude 3.5 Sonnet enabled in your region)

### Setup

```bash
# Clone
git clone https://github.com/Feroz-Hatha/multimodal_rag_pdf.git
cd multimodal_rag_pdf

# Python environment
python3.12 -m venv py312_pdf_rag
source py312_pdf_rag/bin/activate
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Edit .env — fill in AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Create data directories
mkdir -p data/uploads data/processed data/chroma_db

# Start FastAPI
uvicorn src.api.main:app --reload --port 8000

# In a separate terminal — start React dev server
cd web
npm install
npm run dev         # → http://localhost:5173
```

The Vite dev server proxies `/api` to `localhost:8000`, so no CORS configuration is needed.

---

## EC2 Deployment

Requires an EC2 instance (t3.large recommended) with:
- Ubuntu 22.04 LTS
- IAM role with `AmazonBedrockFullAccess`
- Security group: port 22 (SSH), port 80 (HTTP)

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<ELASTIC-IP>

# Clone and run setup script (installs all dependencies, builds React, configures Nginx + systemd)
git clone https://github.com/Feroz-Hatha/multimodal_rag_pdf.git PDF_Multimodal_RAG
cd PDF_Multimodal_RAG
bash deploy/setup_ec2.sh

# Create .env (no AWS keys needed — IAM role handles auth)
cp .env.example .env
nano .env   # set AWS_REGION

# Start
sudo systemctl start pdf-rag
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Upload PDF, returns `job_id` |
| `GET` | `/api/v1/jobs/{job_id}` | Poll indexing progress |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + all chunks |
| `POST` | `/api/v1/query` | Query (non-streaming) |
| `POST` | `/api/v1/query/stream` | Query with SSE streaming |
| `GET` | `/api/v1/health` | Health check + chunk count |

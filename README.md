# BAMS Embedding Service

This repo is a small FastAPI service that turns text into embedding vectors for [BAMS](https://github.com/beysix/Beysix-Wms) (our Brand Asset Management System).

BAMS is where teams track branded physical assets — tents, stages, coolers, signage, whatever gets moved between warehouses and sent out on activations. A single brand might have hundreds of items with messy names, partial descriptions, and photos. People search for things like "something Tusker-branded for an outdoor festival" rather than an exact SKU.

That is where embeddings come in.

## Why we need this

Keyword search only gets you so far. If someone types "outdoor tent" but the asset is logged as "3x3m branded gazebo — green", a plain text match might miss it.

An embedding model reads the text and outputs a list of numbers (a vector) that captures meaning. Similar descriptions end up close together in that number space. We store those vectors in Postgres with pgvector, then use them for:

- **Semantic search** — find assets and activations by intent, not just exact words
- **The in-app assistant** — retrieve relevant chunks of BAMS data before answering a question
- **Future batch jobs** — grouping similar assets, surfacing things that are often used together, building recommendations overnight instead of on every click

The model we run here is [BGE Small](https://huggingface.co/BAAI/bge-small-en-v1.5) (`BAAI/bge-small-en-v1.5`). It is small enough to run on CPU, good enough for English asset descriptions, and outputs **384-dimensional** vectors.

## Why a separate service

We could load the model inside the main Django app, but that app already does a lot: multi-tenant schemas, activations, warehousing, finance, WhatsApp hooks, Celery workers. Pulling in `sentence-transformers` and a few hundred MB of model weights on every web dyno would slow cold starts and make deploys heavier for no good reason.

So this runs on its own. BAMS calls it over HTTP when it needs a vector. If we swap models later, we change this repo — not the whole backend.

```
  BAMS (Django on Render)
        │
        │  POST /embed  { "text": "..." }
        ▼
  This service
        │
        │  BGE Small (loaded once at startup)
        ▼
  { "embedding": [0.024, -0.18, ...] }
        │
        ▼
  Stored in Postgres / pgvector on the BAMS side
```

Most of the heavy embedding work can also run in nightly jobs: find assets that changed today, batch-embed them, write vectors, move on. Users are not waiting on model inference during a normal page load.

## How it works

1. **Startup** — `app.py` loads `BAAI/bge-small-en-v1.5` into memory once. First run downloads the weights from Hugging Face; after that they are cached on disk.

2. **Request** — BAMS (or a workflow script) sends JSON with a `text` field.

3. **Encode** — `sentence-transformers` runs the text through the model with `normalize_embeddings=True` so vectors are unit length. That makes cosine similarity straightforward downstream.

4. **Response** — JSON with an `embedding` array of 384 floats.

For bulk work there is also `POST /embed/batch`, which accepts a `texts` array and returns all vectors in one round trip. Much faster than embedding items one by one during a nightly sync.

### Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/health` | — | `{ "status": "ok", "model": "..." }` |
| `POST` | `/embed` | `{ "text": "..." }` | `{ "embedding": [...] }` |
| `POST` | `/embed/batch` | `{ "texts": ["...", "..."] }` | `{ "embeddings": [[...], [...]] }` |

### Calling it from BAMS

```javascript
const res = await fetch("https://your-embedding-service.onrender.com/embed", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
});

const { embedding } = await res.json();
```

On the BAMS side, point vector search at this service and set `VECTOR_INDEX_EMBEDDING_DIMENSIONS=384` so pgvector column sizes match BGE Small.

## Run it locally

```bash
cd bams-embedding-service
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

First startup takes a minute while the model downloads.

**Single text:**

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Tusker branded outdoor tent"}'
```

**Batch:**

```bash
curl -X POST http://localhost:8000/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Tusker branded outdoor tent","Festival stage backdrop"]}'
```

**Health:**

```bash
curl http://localhost:8000/health
```

## Deploy on Render

Workflow we use:

1. Push this repo to GitHub (its own repo — not bundled inside the main BAMS monorepo deploy).
2. Create a **Web Service** on Render from that repo.
3. Start command:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

   Or deploy with the included `Dockerfile` / `render.yaml`.

4. Optional env var: `EMBEDDING_MODEL` (defaults to `BAAI/bge-small-en-v1.5`).

No GPU required. A Starter instance on CPU is fine for our volume — think a few hundred embeddings per minute, more if you batch.

Point `VECTOR_INDEX_EMBEDDING_BASE_URL` (or whatever URL BAMS uses for embeddings in production) at the Render service URL once it is live.

## What's in the repo

```
app.py              # FastAPI app + model loading
requirements.txt    # fastapi, uvicorn, sentence-transformers
Dockerfile          # container build
render.yaml         # Render blueprint
```

## Model reference

| | |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` |
| Dimensions | 384 |
| Normalization | L2-normalized at encode time |
| Override | set `EMBEDDING_MODEL` env var |

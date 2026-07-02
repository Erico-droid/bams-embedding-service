# BAMS Embedding Service

A dedicated FastAPI microservice that serves text embeddings for BAMS using [BGE Small](https://huggingface.co/BAAI/bge-small-en-v1.5). This service runs separately from the main BAMS API so the embedding model stays isolated, easy to scale, and simple to swap later.

## How to run an embedding model on Render

This is one of the places where Render Workflows can be very useful.

There are three ways to run embeddings on Render, and I recommend one over the others for BAMS.

### Option 1 (Recommended): Run an embedding API as a separate service

Instead of embedding inside your main BAMS server, create a dedicated microservice.

```
BAMS API
     │
     │ HTTP
     ▼
Embedding Service
     │
     ▼
BGE Model
```

Your application simply calls:

```http
POST /embed
```

and gets back:

```json
{
  "embedding": [0.024, -0.18, ...]
}
```

This keeps the embedding model isolated and makes it easy to scale or swap models later.

#### Step 1

Create a new Render Web Service.

Python is the easiest choice because the embedding ecosystem is excellent.

Install:

```bash
pip install fastapi
pip install uvicorn
pip install sentence-transformers
```

#### Step 2

Load the model once at startup.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)
```

This downloads the model the first time. After that it's cached.

#### Step 3

Create an endpoint.

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/embed")
def embed(request: dict):

    vector = model.encode(
        request["text"],
        normalize_embeddings=True
    )

    return {
        "embedding": vector.tolist()
    }
```

Done.

#### Step 4

Call it from BAMS.

```javascript
const response = await fetch(
    "https://embeddings.yourdomain.com/embed",
    {
        method: "POST",
        body: JSON.stringify({
            text
        }),
        headers: {
            "Content-Type": "application/json"
        }
    }
);

const embedding = await response.json();
```

### Option 2: Run it inside your existing backend

If your backend is Node:

```
Node API
 ├── Assets
 ├── Users
 ├── AI
 └── Python child process
```

Every time you need an embedding:

```
Node
   ↓
Python
   ↓
Model
   ↓
Embedding
```

This works but becomes harder to maintain as traffic grows.

### Option 3: Nightly Workflow (my favorite for BAMS)

Since you already plan to run AI jobs every night:

```
00:00
   │
Workflow starts
   │
Find changed assets
   │
Generate embeddings
   │
Save vectors
   │
Generate categories
   │
Generate "works well together"
   │
Generate recommendations
   │
Done
```

No user waits for embeddings. Everything is precomputed.

## Hardware

You do not need a GPU.

Models like:

- BGE Small
- MiniLM
- Nomic Embed

run perfectly on CPU.

Even a Render Starter instance can usually generate a few hundred embeddings per minute, depending on the model and hardware.

## My recommendation for BAMS

I'd build it like this:

```
                 Render

        ┌────────────────────┐
        │     BAMS API        │
        └─────────┬───────────┘
                  │
                  │
                  ▼
        ┌────────────────────┐
        │ Embedding Service  │
        │ BGE Small          │
        └─────────┬───────────┘
                  │
                  ▼
             PostgreSQL
             pgvector

Every Midnight
      │
      ▼
Render Workflow
      │
      ├── Embed new assets
      ├── Cluster by similarity
      ├── Generate AI categories
      ├── Find assets used together
      ├── Update recommendation tables
      └── Finish
```

This architecture has a few advantages:

- Your main API stays responsive because it never loads the embedding model.
- The embedding service can be reused anywhere in BAMS (search, recommendations, AI categorization).
- Nightly jobs handle most of the heavy work, so users aren't waiting on expensive computations.
- If you later decide to switch from BGE to another model, you only change the embedding service without touching the rest of your application.

For your expected workload, this is a clean and scalable design. You can also batch multiple texts into a single request (`model.encode([...])`) during the nightly workflow, which is much faster than embedding assets one at a time.

## Do I create the embedding service on my localhost then push it to my server?

Yes — exactly.

Create it locally first, test it, then push it to GitHub and deploy it as a separate Render Web Service.

```
Local machine
  ↓
Create FastAPI embedding service
  ↓
Test /embed locally
  ↓
Push to GitHub
  ↓
Create Render Web Service from repo
  ↓
BAMS API calls Render embedding URL
```

## Project structure

```
bams-embedding-service/
  app.py
  requirements.txt
  Dockerfile
  render.yaml
  README.md
```

## Run locally

```bash
cd bams-embedding-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The first startup downloads `BAAI/bge-small-en-v1.5` and caches it locally.

### Test single embedding

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Tusker branded outdoor tent"}'
```

### Test batch embeddings

```bash
curl -X POST http://localhost:8000/embed/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Tusker branded outdoor tent","Festival stage backdrop"]}'
```

### Health check

```bash
curl http://localhost:8000/health
```

## Deploy to Render

1. Push this repository to GitHub.
2. Create a new Render Web Service from the repo.
3. Use Docker or Python with this start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

4. Set `EMBEDDING_MODEL` to `BAAI/bge-small-en-v1.5` if you want to override the default.

Then your main BAMS backend calls:

```javascript
const res = await fetch("https://your-embedding-service.onrender.com/embed", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text }),
});

const { embedding } = await res.json();
```

I'd keep this as a separate service, not inside your main BAMS API. That keeps your main server lighter and easier to scale.

## Model notes

- **Model:** `BAAI/bge-small-en-v1.5`
- **Output dimensions:** 384
- **Normalization:** embeddings are L2-normalized (`normalize_embeddings=True`)

When wiring this into BAMS vector search, set `VECTOR_INDEX_EMBEDDING_DIMENSIONS=384` to match BGE Small.

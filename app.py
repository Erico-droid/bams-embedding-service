import os

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

app = FastAPI(title="BAMS Embedding Service")
model = SentenceTransformer(MODEL_NAME)


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed")
def embed(req: EmbedRequest):
    vector = model.encode(req.text, normalize_embeddings=True)
    return {"embedding": vector.tolist()}


@app.post("/embed/batch")
def embed_batch(req: BatchEmbedRequest):
    vectors = model.encode(req.texts, normalize_embeddings=True)
    return {"embeddings": [vector.tolist() for vector in vectors]}

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [bams-embed] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bams_embedding")

_model = None
_model_error = ""
_model_lock = threading.Lock()
_model_load_started_at = 0.0


def _text_stats(texts):
    if not texts:
        return 0, 0
    char_count = sum(len(str(text or "")) for text in texts)
    return len(texts), char_count


def _load_model():
    global _model, _model_error
    started_at = time.monotonic()
    logger.info("model_load_started model=%s", MODEL_NAME)
    try:
        from sentence_transformers import SentenceTransformer

        loaded = SentenceTransformer(MODEL_NAME)
        latency_ms = int((time.monotonic() - started_at) * 1000)
        with _model_lock:
            _model = loaded
            _model_error = ""
        logger.info(
            "model_load_complete model=%s latency_ms=%d dimensions=%d",
            MODEL_NAME,
            latency_ms,
            int(getattr(loaded, "get_sentence_embedding_dimension", lambda: 0)() or 0),
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - started_at) * 1000)
        with _model_lock:
            _model = None
            _model_error = str(exc)
        logger.exception(
            "model_load_failed model=%s latency_ms=%d error=%s",
            MODEL_NAME,
            latency_ms,
            exc,
        )


def _get_model():
    with _model_lock:
        if _model_error:
            raise HTTPException(
                status_code=503,
                detail=f"Model failed to load: {_model_error}",
            )
        if _model is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        return _model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model_load_started_at
    _model_load_started_at = time.monotonic()
    logger.info(
        "service_starting model=%s log_level=%s port=%s",
        MODEL_NAME,
        LOG_LEVEL,
        os.environ.get("PORT", "8000"),
    )
    loader = threading.Thread(target=_load_model, name="embedding-model-loader", daemon=True)
    loader.start()
    yield
    logger.info("service_shutdown model=%s", MODEL_NAME)


app = FastAPI(title="BAMS Embedding Service", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    started_at = time.monotonic()
    response = await call_next(request)
    latency_ms = int((time.monotonic() - started_at) * 1000)
    logger.info(
        "http_request method=%s path=%s status=%s latency_ms=%d",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


@app.get("/health")
def health():
    with _model_lock:
        if _model_error:
            logger.warning("health_check status=error model=%s", MODEL_NAME)
            return {"status": "error", "model": MODEL_NAME, "detail": _model_error}
        if _model is None:
            loading_ms = int((time.monotonic() - _model_load_started_at) * 1000)
            logger.debug("health_check status=loading model=%s loading_ms=%d", MODEL_NAME, loading_ms)
            return {"status": "loading", "model": MODEL_NAME, "loading_ms": loading_ms}
        logger.debug("health_check status=ok model=%s", MODEL_NAME)
        return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed")
def embed(req: EmbedRequest):
    started_at = time.monotonic()
    text_count, char_count = _text_stats([req.text])
    logger.info("embed_request_started text_count=%d char_count=%d", text_count, char_count)
    vector = _get_model().encode(req.text, normalize_embeddings=True)
    latency_ms = int((time.monotonic() - started_at) * 1000)
    logger.info(
        "embed_request_complete text_count=%d char_count=%d vector_dims=%d latency_ms=%d",
        text_count,
        char_count,
        len(vector),
        latency_ms,
    )
    return {"embedding": vector.tolist()}


@app.post("/embed/batch")
def embed_batch(req: BatchEmbedRequest):
    started_at = time.monotonic()
    text_count, char_count = _text_stats(req.texts)
    logger.info("embed_batch_started text_count=%d char_count=%d", text_count, char_count)
    vectors = _get_model().encode(req.texts, normalize_embeddings=True)
    latency_ms = int((time.monotonic() - started_at) * 1000)
    vector_dims = len(vectors[0]) if len(vectors) else 0
    logger.info(
        "embed_batch_complete text_count=%d char_count=%d vector_dims=%d latency_ms=%d",
        text_count,
        char_count,
        vector_dims,
        latency_ms,
    )
    return {"embeddings": [vector.tolist() for vector in vectors]}

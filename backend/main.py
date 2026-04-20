from __future__ import annotations

import os
import re
import string
import tempfile
import time
import warnings
from contextlib import asynccontextmanager

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

load_dotenv()
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "my_model"

# ── Globals — populated in lifespan ──────────────────────────────────────────
model         = None
vectorizer    = None
model_version = None

# ── Prometheus metrics ────────────────────────────────────────────────────────
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry,
)
PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry,
)


# ── MLflow helpers ────────────────────────────────────────────────────────────

def get_latest_model_version(model_name: str) -> str | None:
    """Get version number of the champion model via alias."""
    client = mlflow.MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, "champion")
        return version.version
    except Exception as e:
        print(f"⚠️ champion alias not found ({e}), falling back to latest version")
        versions = client.get_latest_versions(model_name, stages=["None"])
        return versions[0].version if versions else None


def load_vectorizer_from_mlflow(model_name: str, version: str):
    """
    Download the vectorizer artifact logged alongside this specific
    model version. Guarantees model + vectorizer are always from the
    same training run — prevents feature mismatch errors.
    """
    client = mlflow.MlflowClient()
    mv     = client.get_model_version(model_name, version)
    run_id = mv.run_id

    with tempfile.TemporaryDirectory() as tmp:
        vec_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="vectorizer/vectorizer.pkl",
            dst_path=tmp
        )
        with open(vec_path, "rb") as f:
            return pickle.load(f)


# ── Lifespan — runs once on startup ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer, model_version

    # ── Auth ──────────────────────────────────────────────────────────────
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    os.environ["DAGSHUB_USER_TOKEN"]        = dagshub_token

    # ── MLflow + DagsHub setup ────────────────────────────────────────────
    mlflow.set_tracking_uri(
        "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
    )
    dagshub.init(
        repo_owner="Hello-Mitra",
        repo_name="E2E-Text-Summarization",
        mlflow=True,
    )

    # ── Load model ────────────────────────────────────────────────────────
    # Using mlflow.sklearn.load_model (not pyfunc) so predict_proba
    # is available directly on the sklearn object
    model_version = get_latest_model_version(MODEL_NAME)
    model_uri     = f"models:/{MODEL_NAME}@champion"
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # ── Load matching vectorizer from same MLflow run ─────────────────────
    # Always loaded from MLflow — never from local file — so model and
    # vectorizer are guaranteed to have the same number of features
    vectorizer = load_vectorizer_from_mlflow(MODEL_NAME, model_version)

    print(f"✅ Model '{MODEL_NAME}' version {model_version} loaded successfully")
    print(f"✅ Vectorizer loaded — {len(vectorizer.vocabulary_)} features")

    yield


# ── Text preprocessing ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Apply full preprocessing pipeline — must match training pipeline exactly."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = " ".join(w.lower() for w in text.split())
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = "".join(c for c in text if not c.isdigit())
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Analysis API",
    description="IMDB movie review sentiment classifier — Positive or Negative",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment:  str
    confidence: float
    raw_text:   str
    clean_text: str


@app.get("/health")
def health():
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    return {"status": "ok", "model": MODEL_NAME, "version": model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    clean    = normalize_text(body.text)
    feats    = vectorizer.transform([clean])
    feats_df = pd.DataFrame(
        feats.toarray(),
        columns=[str(i) for i in range(feats.shape[1])]
    )

    prediction = int(model.predict(feats_df)[0])
    sentiment  = "Positive" if prediction == 1 else "Negative"

    # predict_proba works directly since we use mlflow.sklearn.load_model
    try:
        confidence = float(model.predict_proba(feats_df)[0][1])
    except Exception:
        confidence = 1.0 if prediction == 1 else 0.0

    PREDICTION_COUNT.labels(prediction=sentiment).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        raw_text=body.text,
        clean_text=clean,
    )


@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )
from __future__ import annotations

import os
import pickle
import re
import string
import time
import warnings

import dagshub
import mlflow
import mlflow.pyfunc
import pandas as pd
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

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ── MLflow / DagsHub setup ────────────────────────────────────────────────────
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri(
    "https://dagshub.com/Hello-Mitra/E2E-Text-Summarization.mlflow"
)
dagshub.init(
    repo_owner="Hello-Mitra",
    repo_name="E2E-Text-Summarization",
    mlflow=True,
)

# ── Load model from MLflow registry ──────────────────────────────────────────
MODEL_NAME = "my_model"


def get_latest_model_version(model_name: str) -> str | None:
    """Return the latest Production version; fall back to None stage."""
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        versions = client.get_latest_versions(model_name, stages=["None"])
    return versions[0].version if versions else None


model_version = get_latest_model_version(MODEL_NAME)
model_uri     = f"models:/{MODEL_NAME}/{model_version}"
print(f"Loading model from: {model_uri}")
model      = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

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

# ── Text preprocessing ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Apply full preprocessing pipeline to raw input text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Lowercase
    text = " ".join(w.lower() for w in text.split())
    # Remove stopwords
    text = " ".join(w for w in text.split() if w not in stop_words)
    # Remove numbers
    text = "".join(c for c in text if not c.isdigit())
    # Remove punctuation
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Lemmatize
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="IMDB movie review sentiment classifier — Positive or Negative",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment:   str    # "Positive" or "Negative"
    confidence:  float  # probability of positive class
    raw_text:    str
    clean_text:  str


@app.get("/health")
def health():
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    return {"status": "ok", "model": MODEL_NAME, "version": model_version}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start = time.time()

    clean  = normalize_text(body.text)
    feats  = vectorizer.transform([clean])
    feats_df = pd.DataFrame(
        feats.toarray(),
        columns=[str(i) for i in range(feats.shape[1])]
    )

    result     = model.predict(feats_df)
    prediction = int(result[0])
    sentiment  = "Positive" if prediction == 1 else "Negative"

    # Confidence — try predict_proba if available
    try:
        proba      = model.predict_proba(feats_df)
        confidence = float(proba[0][1])
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
    """Expose Prometheus metrics."""
    REQUEST_COUNT.labels(method="GET", endpoint="/metrics").inc()
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )

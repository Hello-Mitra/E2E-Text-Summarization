from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 IMDB Sentiment Analyser")
st.caption("Powered by Logistic Regression + TF-IDF  •  Model served via MLflow")

st.divider()

review = st.text_area(
    "Paste your movie review here:",
    placeholder="e.g. This movie was absolutely brilliant. The acting was superb...",
    height=180,
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

if predict_btn:
    if not review.strip():
        st.warning("Please enter a review before clicking Analyse.")
    else:
        with st.spinner("Analysing sentiment..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"text": review},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                sentiment  = data["sentiment"]
                confidence = data["confidence"]
                clean_text = data["clean_text"]

                st.divider()

                # ── Result card ───────────────────────────────────────────────
                if sentiment == "Positive":
                    st.success(f"### 😊 {sentiment} Review")
                else:
                    st.error(f"### 😞 {sentiment} Review")

                # Confidence bar
                st.metric(
                    label="Confidence (Positive class probability)",
                    value=f"{confidence * 100:.1f}%",
                )
                st.progress(confidence)

                # Preprocessed text
                with st.expander("🔍 See preprocessed text"):
                    st.code(clean_text, language=None)

            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to backend. Make sure it is running.")
            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()
st.caption("Model: Logistic Regression (L1, C=1)  •  Features: TF-IDF  •  Dataset: IMDB 50K reviews")

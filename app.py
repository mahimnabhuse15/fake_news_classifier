import os

# --- Memory optimization for deployment (Render free tier = 512MB) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'          # Reduce memory
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'           # Force CPU only

import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

from ai_detector import analyze_text as detect_ai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_LEN = 500
NUM_WORDS = 5000
MODEL_WEIGHTS_PATH = "model.weights.h5"
TOKENIZER_PATH = "tokenizer.pkl"
BERT_MODEL_PATH = "bert_fakenews_model"

# ---------------------------------------------------------------------------
# Download NLTK data (only runs once)
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Detect which model to use
# ---------------------------------------------------------------------------
USE_BERT = os.path.exists(BERT_MODEL_PATH) and os.path.isdir(BERT_MODEL_PATH)

if USE_BERT:
    # --- Load DistilBERT model ---
    print("🧠 Loading DistilBERT model...")
    from transformers import pipeline as hf_pipeline
    bert_classifier = hf_pipeline(
        "text-classification",
        model=BERT_MODEL_PATH,
        tokenizer=BERT_MODEL_PATH,
        device=-1,  # CPU
    )
    print("✅ DistilBERT loaded successfully!")
    lstm_model = None
    tokenizer = None
else:
    # --- Fallback: Load LSTM model ---
    print("⚠️  DistilBERT model not found. Falling back to LSTM.")
    print("   To upgrade, run train_bert_liar.py on Google Colab.")
    bert_classifier = None

    # Rebuild exact same LSTM architecture used during training
    lstm_model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=64, input_length=MAX_LEN),
        Bidirectional(LSTM(64)),
        Dense(1, activation="sigmoid"),
    ])
    lstm_model.build(input_shape=(None, MAX_LEN))
    lstm_model.load_weights(MODEL_WEIGHTS_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print("✅ LSTM model loaded successfully!")

# ---------------------------------------------------------------------------
# Text preprocessing (for LSTM — mirrors the notebook exactly)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove non-alpha characters, strip stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------
def predict_bert(text: str) -> dict:
    """Predict using DistilBERT."""
    result = bert_classifier(text, truncation=True, max_length=128)[0]
    label = result["label"]  # "Fake" or "Real"
    score = result["score"]

    if label == "Fake":
        fake_prob = score
        real_prob = 1 - score
    else:
        real_prob = score
        fake_prob = 1 - score

    return {
        "label": label,
        "confidence": round(max(fake_prob, real_prob), 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
    }


def predict_lstm(text: str) -> dict:
    """Predict using LSTM."""
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = float(lstm_model.predict(padded, verbose=0)[0][0])

    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(1 - pred, 4),
        "real_probability": round(pred, 4),
    }

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON { "text": "..." } and return multi-signal prediction.
    """
    data = request.get_json(force=True)
    raw_text = data.get("text", "").strip()

    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    # --- Signal 1: Content classification (BERT or LSTM) ---
    if USE_BERT:
        content_result = predict_bert(raw_text)
        model_used = "distilbert"
    else:
        content_result = predict_lstm(raw_text)
        model_used = "lstm"

    # --- Signal 2: AI text detection (statistical) ---
    ai_result = detect_ai(raw_text)

    # --- Signal 3: Ensemble decision ---
    content_label = content_result["label"]
    content_confidence = content_result["confidence"]
    ai_score = ai_result["ai_score"]
    burstiness = ai_result["burstiness"]

    warning = None
    final_label = content_label
    final_confidence = content_confidence

    # --- Case 1: Model says "Real" but AI detection flags patterns ---
    if content_label == "Real" and ai_score >= 0.35:
        # AI-like text classified as Real — suspicious
        penalty = ai_score * 0.4  # up to 40% confidence reduction
        final_confidence = max(0.50, content_confidence - penalty)
        warning = ("This text shows statistical patterns consistent with AI-generated content "
                   "(low burstiness, uniform structure). The confidence has been adjusted. "
                   "Always verify claims with trusted sources.")

    # --- Case 2: Model says "Real" with LOW confidence (< 70%) ---
    # Even without AI detection flag, low confidence = uncertain = warn
    if content_label == "Real" and content_confidence < 0.70:
        final_confidence = min(final_confidence, content_confidence * 0.85)
        if warning:
            warning += " Additionally, the content model has low confidence in this classification."
        else:
            warning = ("The content analysis model has low confidence in this prediction. "
                       "This article may contain misleading information. "
                       "Please verify with trusted fact-checkers.")

    # --- Case 3: Model says "Real" + LOW confidence + AI-like = very suspicious ---
    if content_label == "Real" and content_confidence < 0.70 and ai_score >= 0.30:
        final_confidence = max(0.50, final_confidence * 0.75)
        warning = ("⚠️ Multiple warning signals detected: the content model has low confidence "
                   "AND the text shows AI-generated writing patterns (low burstiness: "
                   f"{burstiness:.2f}, AI score: {ai_score:.0%}). "
                   "This article should be independently verified before trusting.")

    # --- Case 4: Model says "Fake" + AI-like = strong fake signal ---
    if content_label == "Fake" and ai_score >= 0.35:
        final_confidence = min(0.99, content_confidence + 0.05)
        warning = ("This text appears to be AI-generated and contains patterns "
                   "commonly associated with misinformation.")

    return jsonify({
        "label": final_label,
        "confidence": round(final_confidence, 4),
        "fake_probability": content_result["fake_probability"],
        "real_probability": content_result["real_probability"],
        "model_used": model_used,
        "warning": warning,
        "signals": {
            "content_analysis": {
                "label": content_result["label"],
                "confidence": content_result["confidence"],
                "fake_probability": content_result["fake_probability"],
                "real_probability": content_result["real_probability"],
            },
            "ai_detection": {
                "score": ai_result["ai_score"],
                "label": ai_result["ai_label"],
                "burstiness": ai_result["burstiness"],
                "vocab_richness": ai_result["vocab_richness"],
            },
            "linguistic": {
                "avg_sentence_length": ai_result["avg_sentence_length"],
                "sentence_length_std": ai_result["sentence_length_std"],
                "repetition_score": ai_result["repetition_score"],
                "punctuation_diversity": ai_result["punctuation_diversity"],
                "opener_diversity": ai_result["opener_diversity"],
            },
        },
        "disclaimer": "This tool analyzes writing patterns, not factual accuracy. Always verify claims with trusted fact-checkers.",
    })


@app.route("/model-info")
def model_info():
    """Return which model is currently active."""
    return jsonify({
        "model": "distilbert" if USE_BERT else "lstm",
        "bert_available": USE_BERT,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
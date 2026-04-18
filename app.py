import os
import re
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify
import onnxruntime as ort

from ai_detector import analyze_text as detect_ai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_LEN = 500
NUM_WORDS = 5000
MODEL_ONNX_PATH = "model.onnx"
TOKENIZER_PATH = "tokenizer.json"

# ---------------------------------------------------------------------------
# Download NLTK data (only runs once)
# ---------------------------------------------------------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Load ONNX model (lightweight — ~30MB RAM vs ~400MB for TensorFlow)
# ---------------------------------------------------------------------------
print("🧠 Loading ONNX model...")
session = ort.InferenceSession(MODEL_ONNX_PATH)
INPUT_NAME = session.get_inputs()[0].name

# Load tokenizer (JSON — no TensorFlow dependency)
with open(TOKENIZER_PATH, "r") as f:
    tk_data = json.load(f)
    word_index = tk_data["word_index"]
    num_words = tk_data.get("num_words", NUM_WORDS)

print("✅ ONNX model loaded successfully!")

# ---------------------------------------------------------------------------
# Text preprocessing (mirrors the notebook exactly)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove non-alpha characters, strip stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def pad_sequences_manual(seq, maxlen):
    """Pad/truncate sequence to maxlen without needing TensorFlow."""
    result = []
    for s in seq:
        if len(s) >= maxlen:
            result.append(s[:maxlen])
        else:
            result.append([0] * (maxlen - len(s)) + s)
    return np.array(result, dtype=np.float32)


def texts_to_sequences_manual(texts, word_index, num_words):
    """Convert texts to sequences using word_index dict, capping at num_words."""
    sequences = []
    for text in texts:
        words = text.split()
        seq = []
        for word in words:
            index = word_index.get(word)
            # Only include if index exists and is less than num_words limit
            if index is not None and index < num_words:
                seq.append(index)
        sequences.append(seq)
    return sequences

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_lstm(text: str) -> dict:
    """Predict using ONNX model."""
    cleaned = clean_text(text)
    seq = texts_to_sequences_manual([cleaned], word_index, num_words)
    padded = pad_sequences_manual(seq, maxlen=MAX_LEN)

    result = session.run(None, {INPUT_NAME: padded})
    pred = float(result[0][0][0])

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

    # --- Signal 1: Content classification ---
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
        penalty = ai_score * 0.4
        final_confidence = max(0.50, content_confidence - penalty)
        warning = ("This text shows statistical patterns consistent with AI-generated content "
                   "(low burstiness, uniform structure). The confidence has been adjusted. "
                   "Always verify claims with trusted sources.")

    # --- Case 2: Model says "Real" with LOW confidence (< 70%) ---
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
        "model": "lstm-onnx",
        "bert_available": False,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
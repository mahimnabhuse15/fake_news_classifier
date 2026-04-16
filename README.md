# 📰 Fake News Detector — AI-Powered Multi-Signal Analysis

A deep learning web application that classifies news articles as **Real** or **Fake** using a multi-signal approach: content classification, AI-text detection, and linguistic profiling.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Models

### LSTM (Default — ships with the repo)
| Layer | Details |
|-------|---------|
| Embedding | `5000 words → 64 dimensions` |
| Bidirectional LSTM | `64 units` |
| Dense (Output) | `1 unit, sigmoid activation` |
| Training Data | 20K articles (Fake.csv + True.csv) |
| Val Accuracy | ~98.5% |

### DistilBERT (Upgrade — train on Colab)
| Detail | Value |
|--------|-------|
| Base Model | `distilbert-base-uncased` |
| Dataset | LIAR (12.8K fact-checked statements from PolitiFact) |
| Labels | Binary: `pants-fire/false/barely-true` → Fake, `half-true/mostly-true/true` → Real |
| Training | 4 epochs, lr=2e-5, AdamW, early stopping |

## 🔍 Multi-Signal Analysis

The app combines 3 independent signals:

| Signal | What It Does |
|--------|-------------|
| **Content Analysis** | LSTM or DistilBERT classifies content as Fake/Real |
| **AI Detection** | Statistical analysis detects AI-generated text patterns |
| **Linguistic Profiling** | Measures burstiness, vocabulary richness, sentence structure |

The ensemble logic adjusts confidence when signals conflict (e.g., model says "Real" but AI detection flags suspicious patterns).

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fake_news_classifier.git
cd fake_news_classifier
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

Open [http://localhost:5001](http://localhost:5001) in your browser.

## 🧠 Upgrade to DistilBERT (Recommended)

The LSTM model works out of the box but has known limitations with AI-generated text. To upgrade:

1. Open `train_bert_liar.py` in Google Colab (with GPU runtime)
2. Run the script — takes ~10 minutes
3. Download the `bert_fakenews_model/` folder
4. Place it in your project root
5. Restart `python app.py` — it will auto-detect and use BERT

## 📁 Project Structure

```
fake_news_classifier/
├── app.py                  # Flask backend + multi-signal API
├── ai_detector.py          # Statistical AI text detection
├── templates/
│   └── index.html          # Dark-mode multi-signal UI
├── model.weights.h5        # LSTM weights (default)
├── tokenizer.pkl           # LSTM tokenizer
├── bert_fakenews_model/    # DistilBERT (after training)
├── train_bert_liar.py      # Colab training script
├── fake_news_lstm.ipynb    # Original LSTM training notebook
├── requirements.txt
├── Procfile
├── runtime.txt
├── .gitignore
└── README.md
```

## 🌐 Deploy to Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy** 🚀

## 📝 API Reference

### `POST /predict`

**Request:**
```json
{ "text": "Your news article text here..." }
```

**Response:**
```json
{
  "label": "Real",
  "confidence": 0.50,
  "model_used": "lstm",
  "warning": "⚠️ Multiple warning signals detected...",
  "signals": {
    "content_analysis": { "label": "Real", "confidence": 0.53 },
    "ai_detection": { "score": 0.35, "label": "Uncertain" },
    "linguistic": { "burstiness": 0.23, "vocab_richness": 0.76 }
  },
  "disclaimer": "This tool analyzes writing patterns, not factual accuracy."
}
```

### `GET /model-info`
Returns which model is active (`distilbert` or `lstm`).

## ⚠️ Limitations

- The LSTM model detects **writing style**, not factual accuracy
- AI-generated fake news with professional tone may fool the content model
- The AI detection module uses statistical heuristics, not a dedicated ML model
- Always verify claims with trusted fact-checkers (PolitiFact, Snopes, FactCheck.org)

---

Built with ❤️ using Deep Learning & Statistical NLP

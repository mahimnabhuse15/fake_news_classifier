# =============================================================================
# 📰 DistilBERT Fake News Classifier — Colab Training Script
# =============================================================================
# HOW TO USE:
#   1. Open Google Colab (colab.research.google.com)
#   2. Create a new notebook
#   3. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
#   4. Copy-paste this entire script into a single cell and run it
#   5. Download the saved model folder from Google Drive
# =============================================================================

# --- Step 1: Install dependencies ---
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "datasets", "scikit-learn", "accelerate"])

# --- Step 2: Imports ---
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Step 3: Download & Load LIAR Dataset ---
print("\n📥 Downloading LIAR dataset...")

# Download LIAR dataset files
import urllib.request

base_url = "https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/"
files = {
    "train": "train2.tsv",
    "val": "val2.tsv",
    "test": "test2.tsv",
}

for split, fname in files.items():
    url = f"{base_url}{fname}"
    if not os.path.exists(fname):
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(url, fname)

# Column names for LIAR-PLUS dataset
columns = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state_info", "party", "barely_true_count", "false_count",
    "half_true_count", "mostly_true_count", "pants_fire_count",
    "context", "justification"
]

# Load TSV files
train_df = pd.read_csv("train2.tsv", sep="\t", header=None, names=columns, on_bad_lines="skip")
val_df = pd.read_csv("val2.tsv", sep="\t", header=None, names=columns, on_bad_lines="skip")
test_df = pd.read_csv("test2.tsv", sep="\t", header=None, names=columns, on_bad_lines="skip")

print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"  Labels: {train_df['label'].unique()}")

# --- Step 4: Binarize Labels ---
# Fake: pants-fire, false, barely-true
# Real: half-true, mostly-true, true
label_map = {
    "pants-fire": 0,     # Fake
    "false": 0,          # Fake
    "barely-true": 0,    # Fake
    "half-true": 1,      # Real
    "mostly-true": 1,    # Real
    "true": 1,           # Real
}

def preprocess_df(df):
    """Clean and binarize labels."""
    df = df[["statement", "label"]].copy()
    df = df.dropna()
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(label_map.keys())]
    df["label"] = df["label"].map(label_map)
    df = df.reset_index(drop=True)
    return df

train_df = preprocess_df(train_df)
val_df = preprocess_df(val_df)
test_df = preprocess_df(test_df)

print(f"\n📊 After binarization:")
print(f"  Train: {len(train_df)} (Fake: {(train_df['label']==0).sum()}, Real: {(train_df['label']==1).sum()})")
print(f"  Val:   {len(val_df)} (Fake: {(val_df['label']==0).sum()}, Real: {(val_df['label']==1).sum()})")
print(f"  Test:  {len(test_df)} (Fake: {(test_df['label']==0).sum()}, Real: {(test_df['label']==1).sum()})")

# --- Step 5: Create HuggingFace Datasets ---
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df),
})

# --- Step 6: Tokenize ---
print("\n🔤 Tokenizing with DistilBERT tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["statement"],
        padding="max_length",
        truncation=True,
        max_length=128,  # LIAR statements are short (avg ~20 words)
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["statement"])
tokenized_dataset.set_format("torch")

print(f"  Tokenization complete!")
print(f"  Sample: {tokenizer.decode(tokenized_dataset['train'][0]['input_ids'][:30])}")

# --- Step 7: Fine-tune DistilBERT ---
print("\n🧠 Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "Fake", 1: "Real"},
    label2id={"Fake": 0, "Real": 1},
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_training_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n🚀 Starting training...")
train_result = trainer.train()
print(f"\n✅ Training complete!")
print(f"  Training loss: {train_result.training_loss:.4f}")

# --- Step 8: Evaluate on Test Set ---
print("\n📈 Evaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"  Test F1 Score: {test_results['eval_f1']:.4f}")
print(f"  Test Precision: {test_results['eval_precision']:.4f}")
print(f"  Test Recall: {test_results['eval_recall']:.4f}")

# Detailed classification report
predictions = trainer.predict(tokenized_dataset["test"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
print("\n📋 Classification Report:")
print(classification_report(labels, preds, target_names=["Fake", "Real"]))

# --- Step 9: Save Model & Tokenizer ---
save_path = "bert_fakenews_model"
print(f"\n💾 Saving model to '{save_path}/'...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"  Model saved! Size: {sum(os.path.getsize(os.path.join(save_path, f)) for f in os.listdir(save_path)) / 1e6:.1f} MB")

# --- Step 10: Quick Inference Test ---
print("\n🧪 Quick inference test:")

test_texts = [
    "The president signed a new trade agreement with European nations yesterday.",
    "SHOCKING: Scientists discover that the Earth is actually flat and NASA has been lying!",
    "The unemployment rate dropped to 3.5 percent according to the Bureau of Labor Statistics.",
    "Government secretly implanting microchips through vaccines to control population.",
]

from transformers import pipeline
classifier = pipeline("text-classification", model=save_path, tokenizer=save_path)

for text in test_texts:
    result = classifier(text, truncation=True, max_length=128)[0]
    emoji = "✅" if result["label"] == "Real" else "❌"
    print(f"  {emoji} [{result['label']}] ({result['score']:.3f}) — {text[:70]}...")

# --- Step 11: Download Instructions ---
print("\n" + "="*60)
print("📦 DOWNLOAD INSTRUCTIONS")
print("="*60)
print(f"""
Your trained model is saved in: ./{save_path}/

Option A — Download directly:
  1. In the Colab file browser (left panel), find '{save_path}/'
  2. Right-click → Download each file
  3. Place them in your project: fake_news_classifier/{save_path}/

Option B — Save to Google Drive first:
  Run this code in a new cell:

  from google.colab import drive
  drive.mount('/content/drive')
  !cp -r {save_path} /content/drive/MyDrive/{save_path}

  Then download from Google Drive.

Files you need:
  - config.json
  - model.safetensors
  - tokenizer_config.json
  - vocab.txt
  - special_tokens_map.json
""")
print("✅ All done! Copy the model folder to your project and run `python app.py`")

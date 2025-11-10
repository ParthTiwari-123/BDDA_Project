import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset # Corrected import: 'datasets' not 'sklearn.datasets'

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- Load Data ----------
def load_data(path):
    """
    Loads data from CSV. The CSV must have 'text' and 'label' columns.
    Maps string labels to integers and returns the mapping.
    """
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    
    mapping = None
    if df["label"].dtype == object:
        # Create mapping
        unique_labels = sorted(df["label"].unique())
        mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
        df["label"] = df["label"].map(mapping)
        print("Label mapping created:", mapping)
    
    return df, mapping

# ---------- TF-IDF + Logistic Regression ----------
def train_tfidf(df, mapping, save_path="tfidf_model.joblib"):
    """
    Trains a TF-IDF + Logistic Regression pipeline and saves it,
    including the label mapping.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    
    print("Training TF-IDF + Logistic Regression model...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    print("\n--- TF-IDF Model Report ---")
    print(classification_report(y_test, preds))
    
    # Create reverse mapping (id -> label)
    if mapping:
        reverse_mapping = {i: lbl for lbl, i in mapping.items()}
    else:
        # Handle case where labels were already numeric
        reverse_mapping = {i: str(i) for i in sorted(df['label'].unique())}

    # Save the pipeline and the reverse mapping together
    joblib.dump((pipeline, reverse_mapping), save_path)
    print(f"Saved TF-IDF model and mapping to {save_path}")

# ---------- Transformer (BERT/DistilBERT) ----------
def train_bert(df, mapping, model_name="distilbert-base-uncased", output_dir="bert_model", epochs=3, batch_size=16):
    """
    Fine-tunes a Transformer model and saves it.
    The label mappings are saved in the model's config.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Convert pandas to Hugging Face Dataset
    train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
    test_ds  = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

    # Tokenize
    def tokenize_fn(batch): 
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=128)
        
    train_tok = train_ds.map(tokenize_fn, batched=True)
    test_tok  = test_ds.map(tokenize_fn, batched=True)

    # Define mappings for model config
    if mapping:
        id2label = {i: lbl for lbl, i in mapping.items()}
        label2id = mapping
    else:
        # Handle case where labels were already numeric
        unique_labels = sorted(df['label'].unique())
        id2label = {i: str(i) for i in unique_labels}
        label2id = {str(i): i for i in unique_labels}

    num_labels = len(df["label"].unique())
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,  # Save mapping
        label2id=label2id   # Save mapping
    )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        seed=RANDOM_SEED,
        logging_steps=50,
        metric_for_best_model="accuracy", # Explicitly set metric
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": (preds == p.label_ids).mean()}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\n--- Training BERT model --- (GPU recommended)")
    trainer.train()
    
    print("\n--- BERT Model Evaluation ---")
    metrics = trainer.evaluate()
    print(metrics)
    
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file with text,label")
    parser.add_argument("--model", choices=["tfidf", "bert"], default="tfidf")
    parser.add_argument("--bert_model_name", type=str, default="distilbert-base-uncased", help="Base BERT model to use")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    df, mapping = load_data(args.data)
    
    if args.model == "tfidf":
        train_tfidf(df, mapping)
    else:
        train_bert(
            df, 
            mapping, 
            model_name=args.bert_model_name,
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
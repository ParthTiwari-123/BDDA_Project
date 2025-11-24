import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data(path):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    
    mapping = None
    if df["label"].dtype == object:
        unique_labels = sorted(df["label"].unique())
        mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
        df["label"] = df["label"].map(mapping)
        print("Label mapping created:", mapping)
    
    return df, mapping

def train_tfidf(df, mapping, save_path="tfidf_model.joblib"):
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
    
    if mapping:
        reverse_mapping = {i: lbl for lbl, i in mapping.items()}
    else:
        reverse_mapping = {i: str(i) for i in sorted(df['label'].unique())}

    joblib.dump((pipeline, reverse_mapping), save_path)
    print(f"Saved TF-IDF model and mapping to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file with text,label")
    parser.add_argument("--model", choices=["tfidf"], default="tfidf")
    parser.add_argument("--text", type=str, help="Text to classify")
    args = parser.parse_args()

    df, mapping = load_data(args.data)
    train_tfidf(df, mapping)
    if args.text:
        model, reverse_mapping = joblib.load("tfidf_model.joblib")
        pred = model.predict([args.text])[0]
        readable = reverse_mapping[pred]
        print(f"Prediction: {readable}")
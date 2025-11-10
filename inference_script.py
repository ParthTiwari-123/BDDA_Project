import argparse
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_tfidf(text, model_path):
    """Loads a saved TF-IDF pipeline and predicts on new text."""
    try:
        # Load the tuple (pipeline, reverse_mapping)
        pipeline, reverse_mapping = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Ensure the model file was saved from 'train_updated.py'.")
        return

    # predict expects an iterable
    input_text = [text]
    
    # Get the numeric prediction
    pred_idx = pipeline.predict(input_text)[0]
    
    # Get probabilities
    probs = pipeline.predict_proba(input_text)[0]
    confidence = probs[pred_idx]
    
    # Map numeric index back to string label
    pred_label = reverse_mapping.get(pred_idx, f"Unknown_Index_{pred_idx}")
    
    print("\n--- TF-IDF Prediction ---")
    print(f"Text:       \"{text}\"")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All Probs:  {dict(zip(pipeline.classes_, probs))}")


def predict_bert(text, model_path):
    """Loads a saved Transformer model and predicts on new text."""
    try:
        # Load model and tokenizer from the directory
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from directory {model_path}: {e}")
        print("Ensure this is a valid Hugging Face model directory saved by 'train_updated.py'.")
        return

    # Set model to evaluation mode
    model.eval()

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]
    
    # Get the predicted class index
    pred_idx = torch.argmax(probs).item()
    
    # Get the confidence score
    confidence = probs[pred_idx].item()
    
    # Map index to label string using the model's config
    pred_label = model.config.id2label.get(pred_idx, f"Unknown_Index_{pred_idx}")

    print("\n--- BERT Prediction ---")
    print(f"Text:       \"{text}\"")
    print(f"Prediction: {pred_label}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show all probabilities
    all_probs = {model.config.id2label[i]: prob.item() for i, prob in enumerate(probs)}
    print(f"All Probs:  {all_probs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict text classification")
    parser.add_argument("--model_path", required=True, help="Path to saved model (.joblib file or directory)")
    parser.add_argument("--model_type", choices=["tfidf", "bert"], required=True, help="Type of model to load")
    parser.add_argument("--text", required=True, help="Text to classify")
    
    args = parser.parse_args()

    if args.model_type == "tfidf":
        predict_tfidf(args.text, args.model_path)
    elif args.model_type == "bert":
        predict_bert(args.text, args.model_path)
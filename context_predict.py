# context_predict.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# -----------------------------
# 1) MODEL SETTINGS (Hugging Face)
# -----------------------------
# Hugging Face model repo
MODEL_REPO = "aleksannndra/context-text-classifier"

# Device (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2) Load Model & Tokenizer from Hugging Face
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.to(DEVICE)
model.eval()

# -----------------------------
# 3) Prediction Function
# -----------------------------
def predict(sentence: str):
    """
    Predicts the class of a single sentence.

    Args:
        sentence (str): Input sentence/question

    Returns:
        pred_class (int): 0 or 1
        probs (np.ndarray): Probability distribution over classes
    """
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class = int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return pred_class, probs

# -----------------------------
# 4) Interactive Demo
# -----------------------------
if __name__ == "__main__":
    print("Context Classifier Demo (type 'exit' to quit)")
    while True:
        sentence = input("Enter a question in Polish: ")
        if sentence.lower() == "exit":
            break
        cls, prob = predict(sentence)
        print(f"Predicted class: {cls} | Probabilities: {prob}")

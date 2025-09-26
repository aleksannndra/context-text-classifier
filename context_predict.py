# context_predict.py
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification

# -----------------------------
# 1) Model & Tokenizer from HF
# -----------------------------
MODEL_REPO = "aleksannndra/context-text-classifier"  # public HF repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model directly from Hugging Face
tokenizer = BertTokenizerFast.from_pretrained(MODEL_REPO)
model = BertForSequenceClassification.from_pretrained(MODEL_REPO)
model.to(DEVICE)
model.eval()

# -----------------------------
# 2) Prediction Function
# -----------------------------
def predict(sentence: str):
    """
    Predicts the class of a single sentence.
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
# 3) Interactive Demo
# -----------------------------
if __name__ == "__main__":
    print("Context Classifier Demo (type 'exit' to quit)")
    while True:
        sentence = input("Enter a question in Polish: ")
        if sentence.lower() == "exit":
            break
        cls, prob = predict(sentence)
        print(f"Predicted class: {cls} | Probabilities: {prob}")

# context_predict.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# -----------------------------
# 1) MODEL PATH SETTINGS
# -----------------------------
# Colab: uncomment the next two lines if using Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Update MODEL_PATH depending on where the model is stored
# Example: Google Drive in Colab
# MODEL_PATH = "/content/drive/My Drive/final_best_model"
# Example: Local machine
MODEL_PATH = "/Users/aleksandrakopytek/Documents/BIELIK/Context_classifier/final_best_model"

# Device (GPU if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 2) Check if model folder exists
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model folder not found at {MODEL_PATH}. "
        "Please check the path or download the trained model to this location."
    )

# -----------------------------
# 3) Load Model & Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# -----------------------------
# 4) Prediction Function
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
# 5) Interactive Demo
# -----------------------------
if __name__ == "__main__":
    print("Context Classifier Demo (type 'exit' to quit)")
    while True:
        sentence = input("Enter a question: ")
        if sentence.lower() == "exit":
            break
        cls, prob = predict(sentence)
        print(f"Predicted class: {cls} | Probabilities: {prob}")

# predict.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# --------- SETTINGS ---------
MODEL_PATH = "./ml
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Load Model & Tokenizer ---------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# --------- Prediction Function ---------
def predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_class = int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    return pred_class, probs

# --------- Demo ---------
if __name__ == "__main__":
    while True:
        sentence = input("Enter a question (or 'exit' to quit): ")
        if sentence.lower() == "exit":
            break
        cls, prob = predict(sentence)
        print(f"Predicted class: {cls} | Probabilities: {prob}")


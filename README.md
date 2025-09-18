# context-text-classifier
“Polish question classifier to detect whether a question requires additional context.”

# Context Classifier

This repository contains a **binary text classifier** that predicts whether a question requires additional context (class 0) or not (class 1). It uses the **HerBERT transformer model** fine-tuned on a custom dataset.

---

## Repository Structure

```
context-classifier/
├── train_context_classifier.py # Script to train the model
├── context_predict.py # Script to run predictions on new questions
├── requirements.txt # Python dependencies
├── README.md # This file
└── models/ # Optional: saved trained models
```

> **Note:** The `models/` folder is not included in the repo due to size. See below for instructions on using the trained model.

---

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```
Dependencies include:

transformers

datasets

torch

scikit-learn

pandas

numpy

optuna

evaluate

sacremoses

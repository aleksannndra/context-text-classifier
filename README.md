# Context Classifier

This repository contains a **binary text classifier** that predicts whether a question requires additional context (class 0) or not (class 1). It uses the **HerBERT transformer model** fine-tuned on a custom dataset.

---

## Repository Structure

```
context-classifier/
├── train_context_classifier.py # Script to train the model
├── context_predict.py # Script to run predictions on new questions
├── requirements.txt # Python dependencies
└── README.md # This file
```

> **Note:** The trained model is hosted on Hugging Face. No local model folder is needed.

---

## Requirements

Install the dependencies:

```
pip install -r requirements.txt
```
Dependencies include:

- transformers
- datasets
- torch
- scikit-learn
- pandas
- numpy
- optuna
- evaluate
- sacremoses

## Training the Model

1. Ensure dataset is saved as an Excel file:
```
pytania_sklasyfikowane.xlsx
```

2. Update the path to the dataset in train_context_classifier.py if needed.

3. Run the training script:
```
python train_context_classifier.py
```

- The script includes hyperparameter tuning with Optuna.
- The final model will be saved locally.

## Running Predictions

**The prediction script loads the trained model directly from Hugging Face:**

1. Run the script:
```
python context_predict.py
```
- Enter a question **in Polish**.
- Type 'exit' to quit.

Example output:
```
Enter a question in Polish: What is the capital of Poland?
Predicted class: 1 | Probabilities: [0.02 0.98]
```

## Notes

- The trained model is hosted on Hugging Face; no local download is required.
- Use a GPU for faster training and inference if available.
- Ensure your internet connection is active for predictions, or load a local copy of the model if needed.


## References

- [HerBERT model](https://huggingface.co/allegro/herbert-base-cased) on Hugging Face  
- [Transformers library](https://huggingface.co/docs/transformers)  
- [Optuna hyperparameter optimization](https://optuna.org/)

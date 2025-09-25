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
- The final model will be saved locally or to Google Drive if using Colab.

## Running Predictions

1. Save trained model to a folder:
```
Colab (Google Drive):

/content/drive/My Drive/final_best_model


Local machine:

./models/final_best_model
```

2. Update the MODEL_PATH in context_predict.py accordingly:
```
# Colab example
MODEL_PATH = "/content/drive/My Drive/final_best_model"

# Local example
# MODEL_PATH = "./models/final_best_model"
```

3. Run the prediction script:
```
python context_predict.py
```

- Enter a question interactively.
- Type 'exit' to quit.
- Example output:
```
Enter a question: What is the capital of Poland?
Predicted class: 1 | Probabilities: [0.02 0.98]
```

## Notes

- The model files are not included in the repo due to size limits. You can:

  - Train the model locally with train_context_classifier.py

  - Or save your trained model to Google Drive and point MODEL_PATH there.

- Use a GPU for faster training and inference if available.

## References

- [HerBERT model](https://huggingface.co/allegro/herbert-base-cased) on Hugging Face  
- [Transformers library](https://huggingface.co/docs/transformers)  
- [Optuna hyperparameter optimization](https://optuna.org/)

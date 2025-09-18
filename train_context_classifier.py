# -------------------------
# robust_finetune_herbert.py
# -------------------------

# Mount Google Drive (Colab only)
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies (Colab-specific, not needed in a local script)
!pip install optuna evaluate transformers datasets scikit-learn pandas numpy torch sacremoses

# General settings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevents tokenizer warnings

# Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -------------------------------
# 1) Load and prepare the dataset
# -------------------------------
try:
    df = pd.read_excel("/content/drive/My Drive/Colab Notebooks/context_data/pytania_sklasyfikowane.xlsx")
except FileNotFoundError:
    print("Error: 'pytania_sklasyfikowane.xlsx' not found. Please ensure the file is in the correct directory.")
    exit()  # Exit if file not found

# Keep only relevant columns
df = df[['Question', 'Class']].dropna().reset_index(drop=True)

# Inspect labels before training
print("Original 'Class' column info:")
print(df['Class'].info())
print("Unique values in 'Class':", df['Class'].unique())

# Save unique labels (should be [0, 1])
unique_labels = df['Class'].unique()
num_labels = len(unique_labels)

# Rename column to match Hugging Face Trainer expectations
df.rename(columns={'Class': 'labels'}, inplace=True)

# Ensure labels are integers (important for training)
try:
    df['labels'] = df['labels'].astype(int)
except ValueError:
    print("Error: 'Class' column contains non-integer values. Please clean the data.")
    exit()

# --------------------------
# 2) Train/Test Split
# --------------------------
train_df, test_df = train_test_split(
    df[['Question', 'labels']],
    test_size=0.2,
    random_state=42,
    stratify=df['labels']  # Ensures balanced split across labels
)

# Convert to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

# --------------------------
# 3) Tokenizer & Encoding
# --------------------------
model_name = "allegro/herbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    # Convert raw text to input_ids & attention masks
    return tokenizer(
        batch["Question"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True
    )

# Apply tokenizer to datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# Format datasets for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch",  columns=["input_ids", "attention_mask", "labels"])

# --------------------------
# 4) Model Setup
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,  # 2 classes: 0 and 1
)

# --------------------------
# 5) Hyperparameter Tuning
# --------------------------

# Function to reinitialize model for each trial
def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

# Define search space for Optuna
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",           # Evaluate at the end of each epoch
    save_strategy="epoch",           # Save checkpoint each epoch
    load_best_model_at_end=True,     # Keep best model according to metric
    metric_for_best_model="f1",
    push_to_hub=False,
    report_to="none",                # Disable logging to TensorBoard
)

# --------------------------
# 6) Metrics
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    labels = np.array(labels)
    preds = np.array(preds)

    # Per-class F1
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0)
    }

    # Save F1 for each class separately
    for i, label_value in enumerate(sorted(unique_labels)):
        if i < len(f1_per_class):
            metrics[f"f1_class_{label_value}"] = f1_per_class[i]
        else:
            metrics[f"f1_class_{label_value}"] = 0.0

    return metrics

# Create Trainer (with Optuna search enabled)
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting hyperparameter search...")

# Run Optuna HPO
import optuna
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=5,  # Small number for quick test, increase for thorough search
)

print("\n--- Hyperparameter Search Complete ---")
print("Best trial number:", best_run.run_id)
print("Best F1 score:", best_run.objective)
print("Best hyperparameters found:", best_run.hyperparameters)

# --------------------------
# 7) Final Training
# --------------------------
print("\n--- Training final model with best hyperparameters ---")

final_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

final_training_args = TrainingArguments(
    output_dir="./final_model_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    report_to="none",
    **best_run.hyperparameters
)

final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

final_trainer.train()

# Save final model to Google Drive
final_trainer.save_model("/content/drive/My Drive/final_best_model")

print("\n Final model saved to /content/drive/My Drive/final_best_model")

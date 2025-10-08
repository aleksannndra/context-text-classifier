# Context-Text-Classifier: Klasyfikator Pytań Konwergentnych

Projekt **Context-Text-Classifier** to dwuklasowy klasyfikator sekwencji oparty na modelu **HerBERT-base-cased**, którego celem jest automatyczne rozróżnianie, czy zadane pytanie w języku polskim wymaga jednoznacznego faktu (np. liczby, daty, nazwy), czy też szerszego kontekstu lub opinii.

Model został wytrenowany i zoptymalizowany przy użyciu **Optuna Hyperparameter Optimization**, osiągając **wysoki wynik F1 (95.9%)** dzięki technikom równoważenia klas oraz świadomemu przygotowaniu zbioru danych.

---

## Definicja problemu

Celem projektu jest **automatyczna klasyfikacja pytań** w języku polskim, np. dla zastosowań w systemach **Question Answering (QA)** lub **Chatbotach**.

| Klasa | Nazwa | Opis | Przykład |
|:---:|---|---|---|
| **1** | Konwergentne / Fakt | Pytanie jednoznaczne, z jedną możliwą odpowiedzią (fakt, liczba, nazwa, data). | „Ile serc ma ośmiornica?” |
| **0** | Dywersyjne / Brak faktu | Pytanie niepełne, subiektywne, zbyt ogólne lub wymagające opisu. | „Ile?” |

---

## Wyniki i metryki

Model został wytrenowany przy użyciu **Optuny (HPO)** oraz **ważonej funkcji strat (Weighted Cross-Entropy Loss)**.  
Najlepszy wynik osiągnięto po **4 epokach** treningu — model uzyskał bardzo wysokie i zrównoważone wyniki pod względem precyzji (Precision) i czułości (Recall).

| Epoka | Training Loss | Validation Loss | Accuracy | F1 (Weighted) | Precision | Recall | F1 Klasy 0 (Dywersyjne) | F1 Klasy 1 (Fakt) |
|:------:|:--------------:|:----------------:|:---------:|:--------------:|:----------:|:--------:|:----------------------:|:----------------:|
| 1 | – | 0.2796 | 0.9456 | 0.9452 | 0.9477 | 0.9456 | 0.9531 | 0.9352 |
| 2 | 0.2150 | 0.2630 | 0.9541 | 0.9539 | 0.9553 | 0.9541 | 0.9601 | 0.9459 |
| 3 | 0.2150 | 0.2313 | 0.9575 | 0.9574 | 0.9578 | 0.9575 | 0.9626 | 0.9507 |
| **4** | **0.0710** | **0.2935** | **0.9592** | **0.9590** | **0.9600** | **0.9592** | **0.9644** | **0.9522** |

**Najlepszy model**: *Epoka 4*  
**Weighted F1 = 0.9590**, **Precision = 0.9600**, **Recall = 0.9592**


---

## Technologie i metody

- **Model bazowy:** [`allegro/herbert-base-cased`](https://huggingface.co/allegro/herbert-base-cased)
- **Framework:** Hugging Face Transformers + PyTorch  
- **Optymalizacja:** [Optuna](https://optuna.org/) (Hyperparameter Search)
- **Metryki:** Accuracy, Weighted F1, Precision, Recall  
- **Dataset:** autorski zbiór pytań sklasyfikowanych jako faktowe / niefaktowe  
- **Tokenizacja:** AutoTokenizer z paddingiem i truncacją do `max_length=128`

---

## Struktura repozytorium
```
context-text-classifier/
├── train_context_classifier.py # Fine-tuning modelu HerBERT z Optuną
├── context_predict.py # Skrypt do inferencji na nowych pytaniach
├── requirements.txt # Zależności Python
└── README.md # Dokumentacja projekt
```

---

## Uruchomienie projektu

### Klonowanie repozytorium

```
git clone https://github.com/aleksannndra/context-text-classifier.git
cd context-text-classifier
```
### Instalacja zależności
```
pip install -r requirements.txt

```

### Trening modelu
```
python train_context_classifier.py

```

### Predykcja
```
python context_predict.py

```

Przykład interaktywnej sesji:
```
Context Classifier Demo (type 'exit' to quit)
Enter a question in Polish: Ile serc ma ośmiornica?
Predicted class: 1 | Probabilities: [0.04, 0.96]
```

## Użyte techniki optymalizacji
- Optuna Hyperparameter Search
→ automatyczny dobór learning_rate, batch_size, num_train_epochs, weight_decay

- Early Stopping & Best Model Selection
→ model z najwyższym f1 zapisany po treningu

- Weighted Loss Function
→ równoważy wpływ klas, zapobiegając biasowi w danych

- Custom Metrics (per class F1)
→ ocena skuteczności osobno dla każdej klasy

## Wymagania systemowe

- Python 3.10+
- GPU (opcjonalnie, ale rekomendowane dla treningu)
- Pakiety (patrz requirements.txt):
```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
optuna>=3.5.0
evaluate>=0.4.2
scikit-learn>=1.3.0
pandas>=2.1.0
numpy>=1.25.0
sacremoses
```

## Model na Hugging Face
[![Hugging Face Model](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/aleksannndra/context-text-classifier)

























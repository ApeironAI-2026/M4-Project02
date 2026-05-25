# Financial NLP Intelligence

### Naive Bayes vs Logistic Regression vs SVM

**Apeiron AI — Boundless Possibilities, Infinite Potential**

---

## Project Overview

Financial institutions process massive amounts of textual information daily, including:

- Financial news
- Market reports
- Analyst opinions
- Investor discussions

This project builds an end-to-end NLP pipeline to analyze financial text and classify sentiment using classical machine learning techniques.

The system compares multiple NLP models and deploys the best-performing model for interactive prediction.

---

## Dataset

Dataset used:

FiQA-2018 Financial Sentiment Dataset

https://huggingface.co/datasets/pauri32/fiqa-2018

The dataset contains financial text samples labeled with sentiment information.

---

## Project Objectives

Build a complete NLP workflow including:

- Text preprocessing
- Tokenization
- Feature representation
- Classical machine learning models
- Honest model evaluation
- Deployment preparation

---

## Project Structure

```text
M4-Project02/
│
├── dataset/
│   └── fiqa_2018/
│
├── notebooks/
│   └── Financial_NLP_Intelligence_FIQA.ipynb
│
├── model/
│   ├── nb_model.pkl
│   ├── lr_model.pkl
│   ├── svm_model.pkl
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── config.json
│
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

## NLP Pipeline

### Block 1 — Text Is Data

Text preprocessing:

- Lowercasing
- Remove punctuation
- Remove special characters
- Stopword removal
- Stemming

---

### Block 2 — Tokenization

Tokenization performed using:

- NLTK Word Tokenizer

---

### Block 3 — Representations

Feature extraction methods:

- Bag of Words
- TF-IDF

---

### Block 4 — Classical Models

Models compared:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machine (SVM)

---

### Block 5 — Honest Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Visualizations

The notebook includes:

- Dataset exploration
- Class distribution
- Confusion matrix
- Model comparison charts

---

## Results

| Model               | Accuracy | Precision | Recall | F1  |
| ------------------- | -------- | --------- | ------ | --- |
| Naive Bayes         | TBD      | TBD       | TBD    | TBD |
| Logistic Regression | TBD      | TBD       | TBD    | TBD |
| SVM                 | TBD      | TBD       | TBD    | TBD |

---

## Deployment

The best-performing model is exported for Streamlit deployment.

Features:

- Input financial text
- Automatic preprocessing
- Sentiment prediction
- Confidence score

Run:

```bash
streamlit run app.py
```

---

## Installation

Clone repository:

```bash
git clone <repository-url>
cd M4-Project02
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Example Prediction

Input:

```text
Tesla shares increased after stronger-than-expected quarterly earnings.
```

Output:

```text
Prediction:
Positive

Confidence:
94.8%
```

---

## Technologies Used

- Python
- Scikit-Learn
- NLTK
- Hugging Face Datasets
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

---

## CV Description

Developed an end-to-end financial NLP intelligence system using TF-IDF and classical machine learning models (Naive Bayes, Logistic Regression, and SVM) with text preprocessing, model comparison, evaluation, and Streamlit deployment.

---

© 2026 Apeiron AI

# Financial NLP Intelligence

## Deep Neural Network (DNN) vs LSTM vs GRU with Streamlit Deployment

**Apeiron AI — Boundless Possibilities, Infinite Potential**

---

# 📌 Project Overview

Financial institutions process massive volumes of textual information every day, including:

* Financial news headlines
* Earnings reports
* Market analysis
* Investor sentiment
* Economic announcements

This project builds a complete **Financial NLP Intelligence Pipeline** capable of analyzing financial text and automatically predicting sentiment using modern Deep Learning architectures.

The system compares:

* Deep Neural Network (DNN)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)

and deploys the best-performing model through an interactive Streamlit web application.

---

# 🎯 Business Problem

Financial markets react rapidly to breaking news and economic reports. Manual analysis is slow and difficult at scale.

This project helps automate:

* Market sentiment monitoring
* Financial news intelligence
* Risk analysis
* Trading signal support
* Automated sentiment classification

using Natural Language Processing (NLP) and Deep Learning.

---

# 🧠 Learning Objectives

By completing this project, you will learn how to:

* Build NLP preprocessing pipelines
* Clean and tokenize financial text
* Create TF-IDF features
* Create tokenized text sequences
* Use word embeddings
* Train Deep Neural Networks
* Train LSTM networks
* Train GRU networks
* Evaluate NLP classification models
* Compare deep learning architectures
* Save trained models for deployment
* Build and deploy a Streamlit application

---

# 📂 Dataset Information

### Dataset Used

Financial Sentiment Analysis Dataset

Dataset Source:

https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis

Alternative Dataset:

FiQA-2018 Financial Sentiment Dataset

https://huggingface.co/datasets/pauri32/fiqa-2018

---

# 📊 Dataset Structure

| Sentence                                              | Sentiment |
| ----------------------------------------------------- | --------- |
| Netflix stock surges after earnings beat expectations | Positive  |
| Market fears increase as inflation rises sharply      | Negative  |
| Federal Reserve keeps interest rates unchanged        | Neutral   |

---

# 🏗️ Project Structure

```text
Financial-NLP-Intelligence/
│
├── data/
│   └── data.csv
│
├── notebooks/
│   └── Financial_NLP_Intelligence.ipynb
│
├── model/
│   ├── best_sentiment_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
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

# ⚙️ NLP Pipeline

## STEP 1 — Data Loading

The dataset is loaded and inspected for:

* Missing values
* Class distribution
* Sample financial headlines
* Sentiment imbalance

---

## STEP 2 — Text Preprocessing

The NLP preprocessing pipeline includes:

* Lowercasing
* Removing punctuation
* Removing special characters
* Stopword removal
* Tokenization
* Whitespace cleanup

---

## STEP 3 — Feature Engineering

Two text representations are created:

### DNN Input

TF-IDF Vectorization

```python
TfidfVectorizer(max_features=5000)
```

### LSTM / GRU Input

Tokenization + Padding

```python
Tokenizer(num_words=10000)
pad_sequences(maxlen=50)
```

These representations allow comparison between traditional feature engineering and sequence-based deep learning approaches.

---

# 🤖 Deep Learning Models

The following deep learning architectures are compared:

| Model | Purpose                                          |
| ----- | ------------------------------------------------ |
| DNN   | Feedforward neural network using TF-IDF features |
| LSTM  | Sequence modeling with memory cells              |
| GRU   | Efficient sequence modeling with gated units     |

---

# 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

Example:

```python
print(classification_report(y_test, y_pred))
```

---

# 📊 Visualizations

The notebook includes:

* Sentiment distribution charts
* Confusion matrices
* Accuracy comparison charts
* F1-score comparison charts
* Training accuracy curves
* Validation accuracy curves
* Training loss curves
* Validation loss curves

---

# 🧪 Example Prediction

### Input

```text
Tesla shares increased after stronger-than-expected quarterly earnings.
```

### Output

```text
Prediction:
POSITIVE

Confidence:
94.8%
```

---

# 🏆 Results

| Model | Accuracy | Precision | Recall | F1-score |
| ----- | -------- | --------- | ------ | -------- |
| DNN   | TBD      | TBD       | TBD    | TBD      |
| LSTM  | TBD      | TBD       | TBD    | TBD      |
| GRU   | TBD      | TBD       | TBD    | TBD      |

---

# 🧠 Key Insights

* DNN provides a strong baseline using TF-IDF features.
* LSTM captures long-term dependencies in financial text.
* GRU often achieves comparable performance with fewer parameters and faster training.
* Sequence-based deep learning models can better capture contextual information than traditional bag-of-words approaches.

---

# 💾 Model Saving

The best-performing model is exported for deployment.

Saved files:

```text
best_sentiment_model.h5
tokenizer.pkl
label_encoder.pkl
tfidf_vectorizer.pkl
config.json
```

---

# 🚀 Streamlit Deployment

The project includes a Streamlit web application for real-time financial sentiment prediction.

### Features

* Interactive financial text input
* Real-time sentiment prediction
* Confidence score display
* Class probability visualization
* Automatic preprocessing
* Deep learning inference

---

# 🖥️ Streamlit Application Features

✅ Positive / Negative / Neutral prediction

✅ Confidence score visualization

✅ Automatic text preprocessing

✅ Deep Learning inference

✅ Dynamic UI alerts

✅ Cached model loading

---

# 📦 Installation

```bash
git clone <repository-url>

cd Financial-NLP-Intelligence

pip install -r requirements.txt
```

---

# 📋 requirements.txt

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
nltk
streamlit
joblib
wordcloud
```

---

# 🧪 Reproducibility

```python
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

---

# 🔮 Future Improvements

Possible future upgrades:

* FinBERT
* BERT
* RoBERTa
* Transformer Encoder Models
* Attention Mechanisms
* Financial Word Embeddings
* Real-Time News APIs
* Docker Deployment
* Cloud Hosting

---

# 🧰 Technologies Used

* Python
* TensorFlow / Keras
* NLTK
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Joblib

---

# 💼 CV / Resume Description

Developed an end-to-end Financial NLP Intelligence system using Deep Learning architectures (DNN, LSTM, and GRU) for financial sentiment analysis, including text preprocessing, tokenization, feature engineering, model comparison, evaluation, visualization, model serialization, and Streamlit deployment.

---

# 👨‍💻 Author

Apeiron AI

---

# 📜 License

This project is for educational and research purposes.

---

© 2026 Apeiron AI

Boundless Possibilities, Infinite Potential
# Financial NLP Intelligence

## Naive Bayes vs Logistic Regression vs SVM with Streamlit Deployment

**Apeiron AI — Boundless Possibilities, Infinite Potential**

---

# 📌 Project Overview

Financial institutions process massive volumes of textual information every day, including:

* Financial news headlines
* Earnings reports
* Market analysis
* Investor sentiment
* Economic announcements

This project builds a complete **Financial NLP Intelligence Pipeline** capable of analyzing financial text and automatically predicting sentiment using classical machine learning algorithms.

The system compares:

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

and deploys the best-performing model through an interactive **Streamlit web application**.

---

# 🎯 Business Problem

Financial markets react rapidly to breaking news and economic reports. Manual analysis is slow and difficult at scale.

This project helps automate:

* Market sentiment monitoring
* Financial news intelligence
* Risk analysis
* Trading signal support
* Automated sentiment classification

using Natural Language Processing (NLP).

---

# 🧠 Learning Objectives

By completing this project, you will learn how to:

* Build NLP preprocessing pipelines
* Clean and tokenize financial text
* Perform TF-IDF vectorization
* Train machine learning classifiers
* Evaluate NLP classification models
* Compare multiple NLP algorithms fairly
* Save trained models for deployment
* Build and deploy a Streamlit application

---

# 📂 Dataset Information

### Dataset Used

**Financial Sentiment Analysis Dataset**

Dataset Source:

https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis

Alternative Dataset:

FiQA-2018 Financial Sentiment Dataset

https://huggingface.co/datasets/pauri32/fiqa-2018

---

# 📊 Dataset Structure

Typical dataset format:

| Sentence                                              | Sentiment |
| ----------------------------------------------------- | --------- |
| Netflix stock surges after earnings beat expectations | Positive  |
| Market fears increase as inflation rises sharply      | Negative  |
| Federal Reserve keeps interest rates unchanged        | Neutral   |

---

# 🏗️ Project Structure

```text
M4-Project02/
│
├── data/
│   └── data.csv
│
├── notebooks/
│   └── Financial_NLP_Intelligence.ipynb
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

# ⚙️ NLP Pipeline

## STEP 1 — Data Loading

The dataset is loaded using pandas and inspected for:

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

Example:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)
```

---

## STEP 3 — Feature Extraction

Text is transformed into numerical vectors using:

* TF-IDF Vectorization

Configuration:

```python
TfidfVectorizer(max_features=5000)
```

TF-IDF helps emphasize important financial keywords while reducing common word influence.

---

# 🤖 Machine Learning Models

The following machine learning algorithms are compared:

| Model                        | Purpose                                |
| ---------------------------- | -------------------------------------- |
| Naive Bayes                  | Fast probabilistic text classification |
| Logistic Regression          | Linear sentiment classification        |
| Support Vector Machine (SVM) | High-dimensional text classification   |

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
* Word frequency analysis
* Confusion matrices
* Accuracy comparison charts
* Precision/Recall/F1-score comparison plots

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

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Naive Bayes         | TBD      | TBD       | TBD    | TBD      |
| Logistic Regression | TBD      | TBD       | TBD    | TBD      |
| SVM                 | TBD      | TBD       | TBD    | TBD      |

---

# 🧠 Key Insights

* TF-IDF performs well for financial sentiment analysis because financial text is sparse and high-dimensional.
* SVM often performs best in NLP classification tasks because it creates strong decision boundaries.
* Logistic Regression provides strong interpretability and stable performance.
* Naive Bayes trains extremely quickly and performs well on smaller datasets.

---

# 💾 Model Saving

The best-performing model is exported for deployment.

Saved files:

```text
best_model.pkl
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
* Clean UI
* Automatic preprocessing
* Dynamic sentiment feedback

---

# ▶️ Running the Streamlit App

Navigate to the Streamlit folder:

```bash
cd streamlit_app
```

Run the application:

```bash
streamlit run app.py
```

Expected Output:

```text
Local URL: http://localhost:8501
```

---

# 🖥️ Streamlit Application Features

The deployed application supports:

✅ Positive / Negative / Neutral prediction
✅ Confidence score visualization
✅ Automatic TF-IDF preprocessing
✅ Dynamic UI alerts
✅ Real-time inference
✅ Cached model loading

---

# 📦 Installation

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

# 📋 requirements.txt

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
streamlit
joblib
wordcloud
```

---

# 🧪 Reproducibility

Random seeds are configured to ensure reproducible results:

```python
np.random.seed(42)
random.seed(42)
```

---

# ⚠️ Important Notes

* Ensure preprocessing used during inference matches training preprocessing.
* Keep model and vectorizer versions synchronized.
* Use stratified train/test splitting to preserve sentiment balance.

---

# 🔮 Future Improvements

Possible future upgrades:

* BERT-based transformers
* FinBERT integration
* Financial word embeddings
* Real-time news API integration
* Attention-based NLP models
* Deep learning sentiment classification
* Docker deployment
* Cloud hosting

---

# 🧰 Technologies Used

* Python
* Scikit-Learn
* NLTK
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Streamlit
* Joblib
* Hugging Face Datasets

---

# 💼 CV / Resume Description

Developed an end-to-end Financial NLP Intelligence system using TF-IDF vectorization and classical machine learning algorithms (Naive Bayes, Logistic Regression, and SVM) for financial sentiment analysis, including preprocessing, model comparison, evaluation, visualization, artifact serialization, and Streamlit deployment.

---

# 👨‍💻 Author

Apeiron AI

---

# 📜 License

This project is for educational and research purposes.

---

© 2026 Apeiron AI
Boundless Possibilities, Infinite Potential
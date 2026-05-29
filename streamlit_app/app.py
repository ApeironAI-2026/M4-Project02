"""
Financial NLP Intelligence Streamlit App
Apeiron AI
"""

import os
import re
import string
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# ---------------------------------------------------
# DOWNLOAD NLTK RESOURCES
# ---------------------------------------------------
nltk.download("stopwords")
nltk.download("punkt")

# ---------------------------------------------------
# DYNAMIC PATH RESOLUTION
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "best_sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "tfidf_vectorizer.pkl"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "config.json"
)

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Financial NLP Intelligence",
    page_icon="📊",
    layout="centered"
)

# ---------------------------------------------------
# LOAD MODEL & VECTORIZER
# ---------------------------------------------------
@st.cache_resource
def load_assets():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer

try:

    model, vectorizer = load_assets()

except Exception as e:

    st.error(f"Error loading model files: {e}")

    st.stop()

# ---------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[%s]' % re.escape(string.punctuation),
        '',
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# ---------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------
st.title("📊 Financial NLP Intelligence")

st.markdown("""
Analyze financial headlines and predict market sentiment using machine learning.

Supported Sentiments:
- Positive 📈
- Negative 📉
- Neutral 😐
""")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
user_text = st.text_area(
    "Enter Financial News or Headline",
    placeholder="Example: Tesla shares surge after strong quarterly earnings..."
)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
if st.button("🔍 Predict Sentiment"):

    if user_text.strip() == "":

        st.warning("Please enter financial text.")

    else:

        # Clean text
        cleaned_text = clean_text(user_text)

        # Vectorize
        vectorized_text = vectorizer.transform(
            [cleaned_text]
        )

        # Predict
        prediction = model.predict(vectorized_text)[0]

        # Confidence Score
        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                vectorized_text
            )[0]

            confidence = max(probabilities)

        # ---------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------
        st.markdown("## Prediction Result")

        if prediction == "Positive":

            st.success(
                f"📈 Sentiment: {prediction}"
            )

        elif prediction == "Negative":

            st.error(
                f"📉 Sentiment: {prediction}"
            )

        else:

            st.info(
                f"😐 Sentiment: {prediction}"
            )

        # Confidence
        if confidence is not None:

            st.metric(
                "Confidence Score",
                f"{confidence * 100:.2f}%"
            )

        # Cleaned text
        st.markdown("### Cleaned Text")

        st.code(cleaned_text)

# ---------------------------------------------------
# SIDEBAR INFORMATION
# ---------------------------------------------------
st.sidebar.title("ℹ️ Model Information")

st.sidebar.markdown("""
### Models Compared
- Naive Bayes
- Logistic Regression
- SVM

### Vectorization
TF-IDF

### NLP Task
Financial Sentiment Analysis
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.caption(
    "Apeiron AI — Boundless Possibilities, Infinite Potential"
)
"""
Financial NLP Intelligence Streamlit App
Apeiron AI
"""

import os
import re
import string
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# ---------------------------------------------------
# DOWNLOAD NLTK RESOURCES
# ---------------------------------------------------

nltk.download("stopwords")
nltk.download("punkt")

try:
    nltk.download("punkt_tab")
except:
    pass

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "best_sentiment_model.h5"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "tokenizer.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "label_encoder.pkl"
)

# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------

MAX_LEN = 50

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Financial NLP Intelligence",
    page_icon="📊",
    layout="centered"
)

# ---------------------------------------------------
# LOAD ASSETS
# ---------------------------------------------------

@st.cache_resource
def load_assets():

    model = load_model(MODEL_PATH)

    tokenizer = joblib.load(TOKENIZER_PATH)

    encoder = joblib.load(ENCODER_PATH)

    return model, tokenizer, encoder


try:

    model, tokenizer, encoder = load_assets()

except Exception as e:

    st.error(f"Error loading model files: {e}")
    st.stop()

# ---------------------------------------------------
# TEXT CLEANING
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
        word
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("📊 Financial NLP Intelligence")

st.markdown("""
Analyze financial headlines and predict market sentiment using Deep Learning.

Supported Sentiments:

- Positive 📈
- Negative 📉
- Neutral 😐
""")

# ---------------------------------------------------
# INPUT
# ---------------------------------------------------

user_text = st.text_area(
    "Enter Financial News or Headline",
    placeholder="Tesla shares surge after strong quarterly earnings..."
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("🔍 Predict Sentiment"):

    if user_text.strip() == "":

        st.warning("Please enter financial text.")

    else:

        cleaned_text = clean_text(user_text)

        sequence = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post"
        )

        probabilities = model.predict(
            padded_sequence,
            verbose=0
        )[0]

        predicted_class = np.argmax(
            probabilities
        )

        prediction = encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = float(
            np.max(probabilities)
        )

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

        st.metric(
            "Confidence Score",
            f"{confidence * 100:.2f}%"
        )

        st.markdown("### Class Probabilities")

        classes = encoder.classes_

        for label, prob in zip(
            classes,
            probabilities
        ):

            st.write(
                f"**{label}:** {prob*100:.2f}%"
            )

        st.markdown("### Cleaned Text")

        st.code(cleaned_text)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("ℹ️ Model Information")

st.sidebar.markdown("""
### Models Compared

- Deep Neural Network (DNN)
- LSTM
- GRU

### Best Model

GRU

### Input Representation

Tokenized Text + Embedding

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
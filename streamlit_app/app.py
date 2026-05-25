import streamlit as st
import pickle

model=pickle.load(
open(
"../model/best_model.pkl",
"rb"
)
)

vectorizer=pickle.load(
open(
"../model/tfidf_vectorizer.pkl",
"rb"
)
)

st.title(
"Financial Sentiment Intelligence"
)

text=st.text_area(
"Enter financial text"
)

if st.button(
"Predict"
):

    vec=vectorizer.transform(
    [text]
    )

    pred=model.predict(
    vec
    )[0]

    st.success(
    f"Prediction: {pred}"
    )
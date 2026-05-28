import os
import pickle
import streamlit as st

# ---------------------------------------------------
# DYNAMIC PATH RESOLUTION
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "model", "tfidf_vectorizer.pkl")

# ---------------------------------------------------
# PAGE CONFIG & ASSET LOADING
# ---------------------------------------------------
st.set_page_config(page_title="Financial Sentiment", page_icon="📊")

@st.cache_resource
def load_assets():
    with open(MODEL_PATH, "rb") as m_file:
        mod = pickle.load(m_file)
    with open(VECTORIZER_PATH, "rb") as v_file:
        vec = pickle.load(v_file)
    return mod, vec

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model or Vectorizer files not found. Check your path directory structure!")
    st.stop()

# ---------------------------------------------------
# UI INTERFACE
# ---------------------------------------------------
st.title("📊 Financial Sentiment Intelligence")
st.markdown("Analyze financial headlines or tweets to extract underlying market sentiment.")

text = st.text_area("Enter financial text:", placeholder="Type financial headline or market news here...")

# ---------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------
if st.button("🔍 Predict Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        # 1. Transform text to numerical vector
        vec_text = vectorizer.transform([text])
        
        # 2. Predict numeric class label (e.g., 0, 1, or 2)
        pred_numeric = int(model.predict(vec_text)[0])
        
        # 3. Map numerical predictions back to human-readable text
        # (Adjust these keys to -1, 0, 1 if your model uses that encoding instead!)
        sentiment_mapping = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE"
        }
        
        pred_label = sentiment_mapping.get(pred_numeric, f"CLASS {pred_numeric}")
        
        # 4. Dynamic visual output feedback
        if "POSITIVE" in pred_label:
            st.success(f"📈 Sentiment Prediction: {pred_label}")
        elif "NEGATIVE" in pred_label:
            st.error(f"📉 Sentiment Prediction: {pred_label}")
        else:
            st.info(f"😐 Sentiment Prediction: {pred_label}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Apeiron AI Lab — Financial NLP Module")
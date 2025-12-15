import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get the directory of the current script for reliable file path resolution
# This is crucial for deployment on cloud platforms.
BASE_DIR = os.path.dirname(__file__)

# ---------------------------
# NLTK Downloads
# ---------------------------
# Note: These need to be handled correctly in your deployment environment (e.g., Docker, build steps)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------------------------
# Load Tokenizer
# ---------------------------
@st.cache_resource
def load_tokenizer():
    # Use BASE_DIR for the tokenizer path
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    try:
        with open(tokenizer_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Error: Tokenizer file not found at {tokenizer_path}")
        st.stop()
        
tokenizer = load_tokenizer()

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_saved_model():
    # Use BASE_DIR for the model path
    saved_model_path = os.path.join(BASE_DIR, "Saved_model")
    try:
        return tf.keras.models.load_model(saved_model_path)
    except Exception as e:
        st.error(f"Error loading model from {saved_model_path}: {e}")
        st.stop()

model = load_saved_model()
maxlen = 150

# ---------------------------
# Text Preprocessing
# ---------------------------
def process_text(text):
    text = str(text).lower()
    # Remove characters that are not lowercase letters or spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove single letters (often artifacts after cleaning)
    text = re.sub(r"\s+[a-z]\s+", " ", text)
    # Collapse multiple spaces into a single space
    text = re.sub(r"\s+", " ", text).strip() # .strip() added for extra robustness
    
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 3
    ]
    return " ".join(tokens)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

st.markdown("""
    This application uses a deep learning model to classify news articles as **Real** or **Fake**. 
    The results are based on the model's prediction and are not a guarantee.
""")

    # ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

user_input = st.text_area("Paste news text here:",height=200)

if st.button("Predict"):
    # FIX: The colon (:) is required here after the 'if' condition.
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        # 1. Preprocessing
        cleaned = process_text(user_input)
        
        # 2. Tokenization and Padding
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=maxlen)

        # 3. Prediction
        pred = model.predict(padded)
        label = np.argmax(pred, axis=1)[0]
        confidence = pred[0][label]

        # 4. Display Results
        if label == 0:
            st.error(f"🚨 Likely Fake (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Likely Real (Confidence: {confidence:.2f})")

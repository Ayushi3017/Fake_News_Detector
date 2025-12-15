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

# ---------------------------
# NLTK Downloads
# ---------------------------
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
    tokenizer_path = os.path.join(os.getcwd(), "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_saved_model():
    saved_model_path = os.path.join(os.getcwd(), "Saved_model")
    return tf.keras.models.load_model(saved_model_path)

model = load_saved_model()
maxlen = 150

# ---------------------------
# Text Preprocessing
# ---------------------------
def process_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+[a-z]\s+", " ", text)
    text = re.sub(r"\s+", " ", text)
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

user_input = st.text_area("Paste news text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        cleaned = process_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=maxlen)

        pred = model.predict(padded)
        label = np.argmax(pred, axis=1)[0]
        confidence = pred[0][label]

        if label == 0:
            st.error(f"🚨 Likely Fake (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Likely Real (Confidence: {confidence:.2f})")



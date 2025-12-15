import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# NLTK SETUP (REQUIRED ON CLOUD)
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# TEXT PREPROCESSING (UNCHANGED LOGIC)
# -----------------------------
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

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# -----------------------------
# LOAD MODEL (SavedModel ONLY)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("saved_model")

model = load_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Paste a news article below and let the model decide.")

user_input = st.text_area("Enter News Text", height=200)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = process_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=150)

        prediction = model.predict(padded)
        label = np.argmax(prediction)

        if label == 0:
            st.error("🚨 This news is likely **FAKE**")
        else:
            st.success("✅ This news is likely **REAL**")

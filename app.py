# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pickle

# Load model
model = tf.keras.models.load_model("saved_model")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# ==============================
# NLTK setup
# ==============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# ==============================
# ==============================
# Text preprocessing
# ==============================
def process_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    words = word_tokenize(text)
    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 3
    ]
    return " ".join(processed_words)

# ==============================
# Streamlit UI
# ==============================
st.title("Fake News Detector 📰")
st.write("Enter a news article below and the model will predict if it's Fake or Real.")

user_input = st.text_area("News Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        processed = process_text(user_input)
        seq = tokenizer.texts_to_sequences([processed])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=150)
        pred = model.predict(padded)
        label_idx = np.argmax(pred, axis=1)[0]
        label = label_encoder.inverse_transform([label_idx])[0]
        st.success(f"Prediction: **{label}**")

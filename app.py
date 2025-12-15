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


from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150)  # maxlen same as training
    return padded

# ==============================
# NLTK setup
# ==============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

from keras.layers import TFSMLayer
import tensorflow as tf

# Load the model
model = TFSMLayer("saved_model", call_endpoint="serve")

# Function to predict
def predict(text_seq_tensor):
    # text_seq_tensor should be preprocessed exactly like training (padded sequence, shape=(1,150))
    preds = model(text_seq_tensor)
    return preds.numpy()  # returns numpy array

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

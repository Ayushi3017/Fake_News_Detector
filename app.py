import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------------------------
# NLTK Downloads (needed in Streamlit cloud)
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------
# Load pre-trained model & tokenizer
# ---------------------------
model = load_model("model.h5")  # make sure this file is in your repo
tokenizer_word_index = pd.read_pickle("tokenizer_word_index.pkl")  # your tokenizer mapping

tokenizer = Tokenizer()
tokenizer.word_index = tokenizer_word_index
maxlen = 150

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------------------------
# Text Preprocessing
# ---------------------------
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter news text below to check if it's Fake or Real.")

user_input = st.text_area("Enter news here:")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_text = process_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=maxlen)
        pred = model.predict(padded)
        label = np.argmax(pred, axis=1)[0]
        confidence = pred[0][label]

        if label == 0:
            st.error(f"❌ Fake News (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {confidence:.2f})")

# ---------------------------
# Optional: Show confusion matrix if you have test data saved
# ---------------------------
if st.checkbox("Show Confusion Matrix (Optional)"):
    y_true = pd.read_pickle("y_test_labels.pkl")  # Save your test labels
    y_pred_probs = model.predict(pd.read_pickle("X_test_padded.pkl"))  # Save your test sequences
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred_labels)

    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'], ax=ax)
    st.pyplot(fig)

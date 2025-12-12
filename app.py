import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# --- Text preprocessing ---
def process_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 3])

# --- Load saved model ---
model = load_model("model.h5")

# --- Load tokenizer saved locally ---
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["dummy"])  # placeholder; we will override word_index
tokenizer.word_index = pd.read_pickle("tokenizer_word_index.pkl")

maxlen = 150

# --- Streamlit UI ---
st.title("📰 Fake News Detector")
st.write("Enter a news article and the model will predict if it is Real or Fake.")

user_input = st.text_area("News Text", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = process_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=maxlen)
        pred = model.predict(padded)
        label = np.argmax(pred)
        confidence = pred[0][label]
        st.success(f"Prediction: {'Fake' if label==0 else 'Real'} (Confidence: {confidence*100:.2f}%)")

import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

# ==============================
# 1. Load model
# ==============================
model = TFSMLayer("saved_model", call_endpoint="serve")

# ==============================
# 2. Load tokenizer and label encoder
# ==============================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ==============================
# 3. Streamlit UI
# ==============================
st.title("Fake News Detector 📰")
st.write("Enter the news article text below and see if it's real or fake!")

user_input = st.text_area("News Text", "")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        # ==============================
        # 4. Preprocess input
        # ==============================
        # Tokenize and pad the input text to match training
        sequences = tokenizer.texts_to_sequences([user_input])
        max_len = 150  # same as used in training
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_len, padding="post"
        )

        # ==============================
        # 5. Make prediction
        # ==============================
        preds = model(tf.convert_to_tensor(padded_seq, dtype=tf.float32))
        pred_class_idx = np.argmax(preds.numpy(), axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_class_idx])[0]

        st.success(f"Prediction: **{pred_label.upper()}**")

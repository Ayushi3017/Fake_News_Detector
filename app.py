import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
import os
from keras.layers import TFSMLayer

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# ==============================
# Load model using TFSMLayer
# ==============================
model = TFSMLayer(MODEL_PATH, call_endpoint="serve")

# ==============================
# Load tokenizer & label encoder
# ==============================
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(
    page_title="🔥 Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.subheader("Check if your news is **REAL** or **FAKE** 🚨")

user_input = st.text_area(
    "Paste your news article here 👇", 
    placeholder="Type or paste the news text...",
    height=200
)

if st.button("🔍 Predict Now!"):
    if not user_input.strip():
        st.warning("⚠️ Oops! You forgot to enter some news text...")
    else:
        with st.spinner("Analyzing your news... 🧠"):
            # Tokenize and pad
            sequences = tokenizer.texts_to_sequences([user_input])
            padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=150, padding="post"
            )

            # Make prediction
            preds = model(tf.convert_to_tensor(padded_seq, dtype=tf.float32))
            pred_class_idx = np.argmax(preds.numpy(), axis=1)[0]
            pred_label = label_encoder.inverse_transform([pred_class_idx])[0]

        if pred_label.lower() == "fake":
            st.error(f"❌ Prediction: **{pred_label.upper()}** — Be careful, this news seems FAKE!")
        else:
            st.success(f"✅ Prediction: **{pred_label.upper()}** — Looks legit!")

st.markdown("---")
st.markdown("💡 *Tip: Always verify news from trusted sources before sharing!*")

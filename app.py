import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------
# STREAMLIT HEADER
# -----------------------------------------
st.title("📰 Fake News Detector — LSTM Model")
st.write("Real-time text classification using your custom dataset hosted on HuggingFace.")


# -----------------------------------------
# LOAD DATA FROM HUGGING FACE
# -----------------------------------------
st.subheader("📥 Loading Dataset...")

Fake = pd.read_csv(
    "https://huggingface.co/datasets/AyushiAyushi3017/fake-news-detector-dataset/resolve/main/Fake.csv"
)
true = pd.read_csv(
    "https://huggingface.co/datasets/AyushiAyushi3017/fake-news-detector-dataset/resolve/main/True.csv"
)

Fake["label"] = 0
true["label"] = 1

Fake.drop(columns=["title", "date", "subject"], inplace=True)
true.drop(columns=["title", "date", "subject"], inplace=True)

News = pd.concat([Fake, true], ignore_index=True)
News.drop_duplicates(inplace=True)

st.success("Dataset Loaded Successfully!")
st.write(News.head())


# -----------------------------------------
# NLTK SETUP
# -----------------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# -----------------------------------------
# TEXT PREPROCESSING FUNCTION
# -----------------------------------------
def process_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+[a-z]\s+", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text)

    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 3
    ]
    return " ".join(processed_words)


st.subheader("🧼 Preprocessing Text...")
texts = list(News["text"])
cleaned_text = [process_text(t) for t in texts]
st.success("Text Preprocessing Complete!")


# -----------------------------------------
# TRAIN-TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    cleaned_text, News["label"], test_size=0.2, random_state=42
)


# -----------------------------------------
# TOKENIZATION & PADDING
# -----------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

maxlen = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)


# -----------------------------------------
# LABEL ENCODING
# -----------------------------------------
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

y_train_hot = tf.keras.utils.to_categorical(y_train_enc)
y_test_hot = tf.keras.utils.to_categorical(y_test_enc)


# -----------------------------------------
# BUILD LSTM MODEL
# -----------------------------------------
input_layer = Input(shape=(maxlen,))
x = Embedding(vocab_size + 1, 100)(input_layer)
x = Dropout(0.5)(x)
x = LSTM(150, return_sequences=True)(x)
x = Dropout(0.5)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation="softmax")(x)

model = Model(input_layer, output_layer)
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

st.subheader("🧠 Training LSTM Model...")
history = model.fit(
    X_train_pad, y_train_hot, epochs=5, validation_data=(X_test_pad, y_test_hot)
)
st.success("Model Training Complete!")


# -----------------------------------------
# TRAINING GRAPHS
# -----------------------------------------
st.subheader("📊 Training Accuracy & Loss")

fig_acc = plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
st.pyplot(fig_acc)

fig_loss = plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Train", "Validation"])
st.pyplot(fig_loss)


# -----------------------------------------
# CONFUSION MATRIX
# -----------------------------------------
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_hot, axis=1)

conf = confusion_matrix(y_true, y_pred)

st.subheader("🟥 Confusion Matrix")

fig_cm = plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, cmap="Reds", fmt="d", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
st.pyplot(fig_cm)


# -----------------------------------------
# LIVE PREDICTOR
# -----------------------------------------
st.subheader("📝 Live Fake News Check")

user_input = st.text_area("Enter news text to check:")

if st.button("Predict"):
    processed = process_text(user_input)
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(pad)[0]

    if np.argmax(pred) == 0:
        st.error("🚨 This looks FAKE!")
    else:
        st.success("✔ This looks REAL!")

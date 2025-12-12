#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-12-12T08:15:39.294Z
"""

import pandas as pd

# ### Load the  Data


Fake=pd.read_csv("Fake.csv")
true=pd.read_csv("True.csv")

Fake['label']=0

Fake

true['label']=1

true

Fake.drop(columns=["title","date","subject"],inplace=True)
true.drop(columns=["title","date","subject"],inplace=True)

Fake.head()

true.head()

News=pd.concat([Fake,true],ignore_index=True)
News

News.info()

News.isnull().sum()

News.duplicated().sum()

News.drop_duplicates(inplace=True)
News.duplicated().sum()

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import zipfile
import os

# --- NLTK Setup and Data Unzipping (Corrected) ---

download_dir = "/kaggle/working/nltk_data/"
os.makedirs(download_dir, exist_ok=True)
nltk.data.path.append(download_dir)

# Download necessary NLTK packages
nltk.download('wordnet', download_dir)
nltk.download('omw-1.4', download_dir)
nltk.download('punkt', download_dir)
nltk.download('stopwords', download_dir)

def unzip_nltk_data(package_name, target_dir):
    zip_path = os.path.join(target_dir, 'corpora', f'{package_name}.zip')
    extract_path = os.path.join(target_dir, 'corpora')

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception:
            pass

unzip_nltk_data('wordnet', download_dir)
unzip_nltk_data('omw-1.4', download_dir)

# Initialize tools globally for the function to use
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# --- Text Preprocessing Function (Refined) ---

def process_text(text):
    # Ensure input is treated as a string
    text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove all non-alphabetic characters (preserving spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 3. Remove single characters and extra white space
    # Remove all single characters from text (e.g., 'a', 'b', 'c')
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    # Remove extra white space from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # 4. Tokenize the text
    words = word_tokenize(text)

    # 5. Lemmatization, Stop Word Removal, and Length Filtering
    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 3 # Filter stop words and short words
    ]

    # 6. Return the tokens joined back into a single string for vectorization
    return " ".join(processed_words)

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import zipfile
import os

# --- NLTK Setup and Data Downloads (Robust to handle different environments) ---

# Define a custom path for downloads, often necessary in environments like Kaggle/Colab
download_dir = "/kaggle/working/nltk_data/"
os.makedirs(download_dir, exist_ok=True)
nltk.data.path.append(download_dir)

# Download all necessary NLTK packages
print("Downloading NLTK resources...")
try:
    nltk.download('wordnet', download_dir)
    nltk.download('omw-1.4', download_dir)
    nltk.download('punkt', download_dir)
    nltk.download('stopwords', download_dir)
    print("NLTK downloads complete.")
except Exception as e:
    print(f"Error during NLTK download: {e}")
    # Proceed, but extraction might still be needed

# Function to manually unzip packages if the environment requires it (like your initial setup)
def unzip_nltk_data(package_name, target_dir):
    zip_path = os.path.join(target_dir, 'corpora', f'{package_name}.zip')
    extract_path = os.path.join(target_dir, 'corpora')

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Unzipped {package_name}.")
        except Exception:
            pass

unzip_nltk_data('wordnet', download_dir)
unzip_nltk_data('omw-1.4', download_dir)

# Initialize global tools after downloads are confirmed
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# --- Text Preprocessing Function (Refined) ---

def process_text(text):
    """
    Cleans, tokenizes, lemmatizes, and removes stop words/short words from text.
    Returns the processed text as a single string.
    """
    # Ensure input is treated as a string
    text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove all non-alphabetic characters (preserving spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 3. Remove single characters and extra white space
    # Remove all single characters from text (e.g., 'a', 'b', 'c')
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    # Remove extra white space from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # 4. Tokenize the text
    words = word_tokenize(text)

    # 5. Lemmatization, Stop Word Removal, and Length Filtering
    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 3 # Filter stop words and short words
    ]

    # 6. Return the tokens joined back into a single string for vectorization
    return " ".join(processed_words)

import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# --- 1. SIMPLIFIED NLTK DOWNLOADS ---
# This uses the default NLTK path, which is usually more reliable for PunktTokenizer.

print("Attempting simplified NLTK downloads...")
# If these files are already downloaded, NLTK will confirm they are up to date.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("NLTK initialization complete.")

# --- 2. Initialize tools globally ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# --- 3. Text Preprocessing Function ---

def process_text(text):
    """
    Cleans, tokenizes, lemmatizes, and removes stop words/short words from text.
    Returns the processed text as a single string.
    """
    # Ensure input is treated as a string
    text = str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove all non-alphabetic characters (preserving spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 3. Remove single characters and extra white space
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # 4. Tokenize the text (The line that caused the error)
    words = word_tokenize(text)

    # 5. Lemmatization, Stop Word Removal, and Length Filtering
    processed_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 3
    ]

    # 6. Return the tokens joined back into a single string for vectorization
    return " ".join(processed_words)

# --- Test the function with a sample ---
sample_text = "The quick brown fox jumps over the lazy dog, 123 times a day!"
print("\nTesting the function:")
print(f"Original: {sample_text}")
print(f"Processed: {process_text(sample_text)}")

x=News.drop('label',axis=1)
y=News.label

texts=list(x['text'])

cleaned_text = [process_text(text) for text in texts]

print(cleaned_text[:10])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cleaned_text, y, test_size=0.2, random_state=42)

!pip install tensorflow

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_idx = tokenizer.word_index  # Corrected syntax for accessing word index
v = len(word_idx)
print("the size of vocab =", v)  # Corrected spacing
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 150
X_train = pad_sequences(X_train,maxlen=maxlen)
X_test = pad_sequences(X_test,maxlen=maxlen)



y.value_counts()

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense,Input,GlobalMaxPooling1D,Dropout
from tensorflow.keras.models import Model
from keras import optimizers
import numpy as np
from tensorflow.keras.optimizers import Adam



inputt=Input(shape=(maxlen,))
learning_rate = 0.0001
x=Embedding(v+1,100)(inputt)
x = Dropout(0.5)(x)
x = LSTM(150,return_sequences=True)(x)
x = Dropout(0.5)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax')(x)

model = Model(inputt, x)

# Define optimizer with specified learning rate
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# -make label encoder to the labels to pass it to the model:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# transform it to be categorical


import tensorflow as tf

y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)

# # train the model:



history = model.fit(X_train, y_train_one_hot, epochs=15, validation_data=(X_test, y_test_one_hot))


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'],color='Purple')
plt.plot(history.history['val_accuracy'],color='Red')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'],color='Purple')
plt.plot(history.history['val_loss'],color='Red')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# # Accuracy:


# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test_one_hot)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_test_one_hot, axis=1)
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Fake', 'Real'], 
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
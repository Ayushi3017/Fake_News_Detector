# 📰 Fake News Detector — Streamlit App

A lightweight NLP-powered web app that classifies news articles as **Real** or **Fake** using an LSTM-based deep learning model. The dataset is hosted on **Hugging Face**, and the app is fully deployable on **Streamlit Cloud**.

---

## 🚀 Features

* End-to-end text preprocessing (cleaning, tokenization, lemmatization)
* LSTM neural network trained on a custom fake news dataset
* Hugging Face dataset loading via direct raw URLs
* Interactive Streamlit UI for predictions
* Real-time confidence scoring and visualization

---

## 📂 Project Structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── notebook.ipynb
```

---

## 🛠 Tech Stack

* **Frontend:** Streamlit
* **NLP:** NLTK
* **Data Processing:** Pandas, NumPy
* **Modeling:** TensorFlow, Keras, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Dataset Hosting:** Hugging Face Datasets

---

## 📦 Dataset

Dataset used for training:
[https://huggingface.co/datasets/AyushiAyushi3017/fake-news-detector-dataset](https://huggingface.co/datasets/AyushiAyushi3017/fake-news-detector-dataset)

Contains two CSV files: `Fake.csv` and `True.csv`.

---

## 🧠 Model Overview

* Tokenizer + padded sequences
* Embedding layer (100 dimensions)
* LSTM layer (150 units)
* Dropout regularization
* Global Max Pooling
* Dense layers with ReLU and Softmax
* Trained for 15 epochs with 80/20 train-test split

---

## ▶️ Running Locally

### 1. Clone the repository

```
git clone https://github.com/<your-username>/<repo>.git
cd <repo>
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```
streamlit run app.py
```

---

## 🌐 Deployment

Fully compatible with **Streamlit Cloud**.
Push your repo → connect Streamlit → deploy.

---

## 📌 Future Enhancements

* Switch to transformer-based models (BERT/DistilBERT)
* Add URL-based scraping and prediction
* Add a REST API endpoint
* Improve UI design and styling

---

## 🤝 Acknowledgements

Built using open-source tools and a custom Hugging Face dataset.

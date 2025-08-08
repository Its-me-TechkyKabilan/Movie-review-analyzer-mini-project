# 🎬 Movie Sentiment Analysis Project 🎯

This project is a machine learning-based sentiment analysis system trained on movie reviews. It classifies reviews as **positive** or **negative** using NLP techniques and various ML models.

---

## 📁 Project Structure

```
movie_sentiment_project/
│
├── sentiment_training_and_saving.py  # Training pipeline & model saving
├── app.py                            # Streamlit app for user input & prediction
├── model.pkl                         # Trained sentiment model
├── vectorizer.pkl                    # TF-IDF vectorizer used for features
├── requirements.txt                  # Required Python packages
├── README.md                         # You're reading it!
└── ...
```

---

## 🔍 Features

- Uses **NLTK's movie review dataset**
- **TF-IDF vectorization** of text
- Trained using **scikit-learn Pipelines**
- Deployable using **Streamlit** UI
- Outputs whether a given movie review is positive or negative

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/movie_sentiment_project.git
cd movie_sentiment_project
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🧠 Model Training (`sentiment_training_and_saving.py`)

- Loads `nltk.corpus.movie_reviews`
- Converts text to TF-IDF vectors
- Splits data into training/testing sets
- Trains a logistic regression classifier
- Saves the model & vectorizer using `pickle`

---

## 🎯 Streamlit App (`app.py`)

- Takes user input (movie review text)
- Preprocesses using the saved vectorizer
- Predicts sentiment using saved model
- Displays the prediction: ✅ Positive / ❌ Negative

---

## 📦 Dependencies

```
nltk
scikit-learn
streamlit
datasets
pickle
```

Make sure to download NLTK data (first run only):

```python
import nltk
nltk.download('movie_reviews')
```

---

## 📷 Screenshot

![App Screenshot](screenshot.png)

---

## ✨ Future Enhancements

- Add support for multiclass classification (neutral, mixed)
- Use advanced models (BERT, LSTM, etc.)
- Improve UI with themes and graph analytics

---

## 🧑‍💻 Author

- **Kabilan S** – [GitHub](https://github.com/Its-me-TechkyKabilan)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

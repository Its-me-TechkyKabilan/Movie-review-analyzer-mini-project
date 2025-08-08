# sentiment_training_and_saving.py

import os
import pickle
import nltk
from nltk.corpus import movie_reviews
from nltk.sentiment import SentimentIntensityAnalyzer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Download necessary NLTK resources
nltk.download('movie_reviews', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# 2. Load NLTK movie_reviews corpus
texts_nltk = []
labels_nltk = []
for category in movie_reviews.categories():
    for fid in movie_reviews.fileids(category):
        texts_nltk.append(" ".join(movie_reviews.words(fid)))
        labels_nltk.append(1 if category == 'pos' else 0)

# 3. Load IMDb dataset
print("Loading IMDb dataset...")
imdb_dataset = load_dataset('imdb')
imdb_train_texts = list(imdb_dataset['train']['text'])
imdb_train_labels = list(imdb_dataset['train']['label'])
imdb_test_texts = list(imdb_dataset['test']['text'])
imdb_test_labels = list(imdb_dataset['test']['label'])
imdb_texts = imdb_train_texts + imdb_test_texts
imdb_labels = imdb_train_labels + imdb_test_labels

# 4. Combine datasets
texts = texts_nltk + imdb_texts
labels = labels_nltk + imdb_labels

# 5. Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 6. Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
        max_features=30000
    )),
    ('clf', LogisticRegression(solver='liblinear', C=1.0, max_iter=2000))
])

# 7. Train the model
print("Training model on combined NLTK + IMDb data...")
pipeline.fit(X_train, y_train)

# 8. Evaluate on the test set
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Initialize VADER for four-way classification rules
sia = SentimentIntensityAnalyzer()

# 10. Save the pipeline and VADER analyzer for the web app
os.makedirs('saved_models', exist_ok=True)
with open('saved_models/hybrid_pipeline.pkl', 'wb') as f:
    pickle.dump((pipeline, sia), f)
print("Saved trained pipeline + VADER to saved_models/hybrid_pipeline.pkl")

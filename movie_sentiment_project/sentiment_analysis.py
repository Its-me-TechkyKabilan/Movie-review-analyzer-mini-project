# sentiment_analysis.py

import nltk
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# 1. Ensure VADER is available
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# 2. Load and prepare the IMDb dataset
dataset = load_dataset('imdb')
texts  = list(dataset['train']['text']) + list(dataset['test']['text'])
labels = list(dataset['train']['label']) + list(dataset['test']['label'])

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 4. Build a TF-IDF + Logistic Regression pipeline
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
pipeline.fit(X_train, y_train)

# 5. Interactive loop: get movie name & review, then predict
print("Enter 'exit' as the movie name to quit.")
while True:
    movie = input("\n🎬 Movie name: ").strip()
    if movie.lower() == 'exit':
        print("Goodbye!")
        break

    review = input("📝 Review text: ").strip()
    if not review:
        print("⚠️  No review entered—please try again.")
        continue

    # 6. Rule-based sentiment logic with VADER
    lower = review.lower()
    if 'not bad' in lower:
        label = "Neutral"
    else:
        scores = sia.polarity_scores(review)
        pos, neg, comp = scores['pos'], scores['neg'], scores['compound']
        if pos > 0 and neg > 0:
            label = "Mixed"
        elif comp >= 0.05:
            label = "Positive"
        elif comp <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

    # 7. Output the result
    print(f"\n🎥 Movie   : {movie!r}")
    print(f"💬 Review  : {review!r}")
    print(f"📊 Sentiment: {label}")

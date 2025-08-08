import streamlit as st
import nltk
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# 1. Ensure VADER is available
nltk.download('vader_lexicon', quiet=True)

@st.cache_resource
def load_model():
    # Load & train pipeline on IMDb data
    dataset = load_dataset('imdb')
    texts = list(dataset['train']['text']) + list(dataset['test']['text'])
    labels = list(dataset['train']['label']) + list(dataset['test']['label'])
    X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1,2),
            max_df=0.9,
            min_df=5,
            max_features=30000
        )),
        ('clf', LogisticRegression(solver='liblinear', C=1.0, max_iter=2000))
    ])
    pipeline.fit(X_train, y_train)
    sia = SentimentIntensityAnalyzer()
    return pipeline, sia

pipeline, sia = load_model()
st.title("🎥 Movie Review Sentiment Analyzer")

# Initialize session storage for entries
if 'records' not in st.session_state:
    st.session_state['records'] = []

# Callback to clear inputs
def clear_inputs():
    st.session_state['movie_input']  = ""
    st.session_state['review_input'] = ""

# 2. Render widgets *after* defining clear_inputs
movie  = st.text_input("Movie name:", key="movie_input", placeholder="Enter the movie title")
review = st.text_area("Review text:", key="review_input", height=150, placeholder="Type your review here")

# 3. Place Analyze and Clear buttons
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Analyze"):
        if not movie or not review:
            st.warning("Please fill in both fields before analyzing.")
        else:
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

            # Display result
            st.markdown(f"**Movie:** {movie}")
            st.markdown(f"**Review:** {review}")
            st.markdown(f"**Sentiment:** {label}")

            # Store entry
            st.session_state['records'].append({
                'Movie': movie,
                'Review': review,
                'Sentiment': label
            })

with col2:
    # Clear button uses on_click to reset keys before rerun
    st.button("Clear", on_click=clear_inputs)

# 4. Show recorded entries
if st.session_state['records']:
    st.subheader("📊 Session Records")
    df = pd.DataFrame(st.session_state['records'])
    st.table(df)

#.venv\Script\activate.bat
#streamlit run app.py
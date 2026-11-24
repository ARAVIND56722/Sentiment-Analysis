import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

import nltk

_LOCAL_NLTK_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "nltk_data"))
if _LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from nltk.corpus import twitter_samples
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from preprocessor import TweetPreprocessor


def build_3class_dataset() -> pd.DataFrame:
    pos_tweets = twitter_samples.strings("positive_tweets.json")
    neg_tweets = twitter_samples.strings("negative_tweets.json")
    stream_tweets = twitter_samples.strings("tweets.20150430-223406.json")

    sia = SentimentIntensityAnalyzer()
    neutral_candidates = []

    for tweet in stream_tweets:
        score = sia.polarity_scores(tweet)
        if -0.05 <= score["compound"] <= 0.05 and score["pos"] < 0.08 and score["neg"] < 0.08:
            neutral_candidates.append(tweet)
            if len(neutral_candidates) >= 5000:
                break

    records = []
    for t in pos_tweets:
        records.append({"text": t, "sentiment": "positive"})
    for t in neg_tweets:
        records.append({"text": t, "sentiment": "negative"})
    for t in neutral_candidates:
        records.append({"text": t, "sentiment": "neutral"})

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def train_sentiment_model(
    output_dir: str = "models",
) -> Tuple[Pipeline, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    df = build_3class_dataset()

    preprocessor = TweetPreprocessor(spacy_model=None)
    df["clean_text"] = df["text"].apply(lambda x: preprocessor.clean_for_ml(x, lemmatize=False))
    df = df[df["clean_text"].str.strip().str.len() > 1].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"]).tolist()

    model_path = os.path.join(output_dir, "sentiment_pipeline.joblib")
    metrics_path = os.path.join(output_dir, "metrics.joblib")

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "classes": list(pipeline.classes_),
        "sample_count": len(df),
    }

    joblib.dump(pipeline, model_path)
    joblib.dump(metrics, metrics_path)

    return pipeline, metrics


if __name__ == "__main__":
    train_sentiment_model()

import os
import pytest
import pandas as pd
from preprocessor import TweetPreprocessor
from analyzer import TwitterSentimentAnalyzer


@pytest.fixture(scope="module")
def preprocessor():
    return TweetPreprocessor(spacy_model="en_core_web_sm")


@pytest.fixture(scope="module")
def analyzer():
    return TwitterSentimentAnalyzer()


class TestTweetPreprocessor:
    def test_contraction_expansion(self, preprocessor):
        text = "I don't think she'll come, but she can't say no."
        expanded = preprocessor.expand_contractions(text)
        assert "do not" in expanded
        assert "she will" in expanded
        assert "cannot" in expanded

    def test_clean_for_vader_removes_handles_and_urls(self, preprocessor):
        text = "Hello @elonmusk check this out http://example.com/test #awesome"
        cleaned = preprocessor.clean_for_vader(text)
        assert "@elonmusk" not in cleaned
        assert "http" not in cleaned
        assert "awesome" in cleaned

    def test_clean_for_vader_preserves_emojis_and_caps(self, preprocessor):
        text = "This is AMAZING! 😍🚀"
        cleaned = preprocessor.clean_for_vader(text)
        assert "AMAZING" in cleaned
        assert "!" in cleaned
        assert "love" in cleaned or "rocket" in cleaned

    def test_clean_for_ml(self, preprocessor):
        text = "Lovingggg this product so much!!! #happy"
        cleaned = preprocessor.clean_for_ml(text, lemmatize=False)
        assert "loving" in cleaned or "love" in cleaned
        assert "#" not in cleaned

    def test_metadata_extraction(self, preprocessor):
        text = "Hey @user1 and @user2 check out #ai and #nlp! https://t.co/xyz 🔥"
        meta = preprocessor.extract_metadata(text)
        assert set(meta["mentions"]) == {"@user1", "@user2"}
        assert set(meta["hashtags"]) == {"ai", "nlp"}
        assert len(meta["urls"]) == 1
        assert "🔥" in meta["emojis"]
        assert meta["word_count"] > 0


class TestVaderSentimentAnalysis:
    def test_positive_tweet(self, analyzer):
        tweet = "I absolutely love this new feature! It makes my day so wonderful. ❤️"
        res = analyzer.analyze_vader(tweet)
        assert res["sentiment"] == "positive"
        assert res["compound_score"] > 0.3
        assert res["confidence"] > 0.6

    def test_negative_tweet(self, analyzer):
        tweet = "Worst customer service ever. Terrible experience and rude staff! 😡"
        res = analyzer.analyze_vader(tweet)
        assert res["sentiment"] == "negative"
        assert res["compound_score"] < -0.3
        assert res["confidence"] > 0.6

    def test_neutral_tweet(self, analyzer):
        tweet = "The scheduled maintenance will occur at 2:00 AM UTC on Tuesday."
        res = analyzer.analyze_vader(tweet)
        assert res["sentiment"] == "neutral"
        assert -0.05 <= res["compound_score"] <= 0.05


class TestMLSentimentAnalysis:
    def test_ml_model_prediction(self, analyzer):
        tweet = "This is the best phone I have ever owned! Fantastic build."
        res = analyzer.analyze_ml(tweet)
        assert res["sentiment"] in ("positive", "neutral", "negative")
        assert "probabilities" in res
        assert 0.0 <= res["confidence"] <= 1.0


class TestEnsembleAnalysis:
    def test_ensemble_consistency(self, analyzer):
        tweet = "Outstanding performance by the entire team today! Very proud! 🏆"
        res = analyzer.analyze(tweet, method="ensemble")
        assert res["sentiment"] == "positive"
        assert res["confidence"] >= 0.6
        assert "vader_prediction" in res
        assert "ml_prediction" in res


class TestBatchProcessing:
    def test_batch_dataframe(self, analyzer):
        tweets = [
            "Great product and fast delivery!",
            "Terrible app, crashes constantly.",
            "Meeting starts at 10 AM.",
        ]
        df = analyzer.analyze_batch(tweets, text_column="tweet", method="vader")
        assert len(df) == 3
        assert list(df["sentiment"]) == ["positive", "negative", "neutral"]
        assert "confidence" in df.columns
        assert "compound_score" in df.columns

        metrics = analyzer.get_summary_metrics(df)
        assert metrics["total_tweets"] == 3
        assert metrics["sentiment_counts"]["positive"] == 1
        assert metrics["sentiment_counts"]["negative"] == 1
        assert metrics["sentiment_counts"]["neutral"] == 1


class TestEdgeCases:
    def test_empty_string(self, analyzer):
        res = analyzer.analyze("", method="vader")
        assert res["sentiment"] == "neutral"

    def test_all_emojis(self, analyzer):
        res_pos = analyzer.analyze("😍😍😍❤️❤️❤️", method="vader")
        assert res_pos["sentiment"] == "positive"

        res_neg = analyzer.analyze("😡😡😡🤬🤬🤬", method="vader")
        assert res_neg["sentiment"] == "negative"

    def test_urls_and_mentions_only(self, analyzer):
        res = analyzer.analyze("@user1 @user2 http://example.com/test", method="vader")
        assert res["sentiment"] in ("neutral", "positive", "negative")

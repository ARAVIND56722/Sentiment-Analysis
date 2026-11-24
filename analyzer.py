import os
from typing import Dict, Any, List, Union, Optional
import joblib
import pandas as pd
import numpy as np
import nltk

_LOCAL_NLTK_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "nltk_data"))
if _LOCAL_NLTK_DATA not in nltk.data.path:
    nltk.data.path.insert(0, _LOCAL_NLTK_DATA)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from preprocessor import TweetPreprocessor


class TwitterSentimentAnalyzer:
    def __init__(
        self,
        model_path: str = "models/sentiment_pipeline.joblib",
        spacy_model: Optional[str] = "en_core_web_sm",
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
    ):
        self.preprocessor = TweetPreprocessor(spacy_model=spacy_model)
        self.vader = SentimentIntensityAnalyzer()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.model_path = model_path
        self._ml_pipeline = None
        self._ml_metrics = None

    @property
    def ml_pipeline(self):
        if self._ml_pipeline is None:
            if os.path.exists(self.model_path):
                try:
                    self._ml_pipeline = joblib.load(self.model_path)
                except Exception:
                    self._ml_pipeline = None
            else:
                self._ml_pipeline = None
        return self._ml_pipeline

    def train_model(self):
        from model_trainer import train_sentiment_model
        pipeline, metrics = train_sentiment_model()
        self._ml_pipeline = pipeline
        self._ml_metrics = metrics
        return metrics

    def analyze_vader(self, raw_tweet: str) -> Dict[str, Any]:
        cleaned = self.preprocessor.clean_for_vader(raw_tweet)
        text_to_eval = cleaned if cleaned else raw_tweet

        scores = self.vader.polarity_scores(text_to_eval)
        compound = scores["compound"]

        if compound >= self.positive_threshold:
            sentiment = "positive"
            confidence = round(min(1.0, 0.5 + (compound / 2.0)), 3)
        elif compound <= self.negative_threshold:
            sentiment = "negative"
            confidence = round(min(1.0, 0.5 + (abs(compound) / 2.0)), 3)
        else:
            sentiment = "neutral"
            confidence = round(scores["neu"], 3)

        return {
            "method": "vader",
            "sentiment": sentiment,
            "confidence": confidence,
            "compound_score": compound,
            "breakdown": {
                "positive": round(scores["pos"], 3),
                "neutral": round(scores["neu"], 3),
                "negative": round(scores["neg"], 3),
            },
            "cleaned_text": cleaned,
        }

    def analyze_ml(self, raw_tweet: str) -> Dict[str, Any]:
        pipeline = self.ml_pipeline
        if pipeline is None:
            res = self.analyze_vader(raw_tweet)
            res["method"] = "vader_fallback"
            return res

        cleaned = self.preprocessor.clean_for_ml(raw_tweet, lemmatize=False)
        if not cleaned.strip():
            cleaned = raw_tweet.lower()

        classes = list(pipeline.classes_)
        proba = pipeline.predict_proba([cleaned])[0]
        pred_idx = np.argmax(proba)
        pred_sentiment = classes[pred_idx]
        confidence = float(proba[pred_idx])

        proba_dict = {cls_name: round(float(p), 3) for cls_name, p in zip(classes, proba)}

        return {
            "method": "ml",
            "sentiment": pred_sentiment,
            "confidence": round(confidence, 3),
            "probabilities": proba_dict,
            "cleaned_text": cleaned,
        }

    def analyze_ensemble(self, raw_tweet: str) -> Dict[str, Any]:
        vader_res = self.analyze_vader(raw_tweet)
        ml_res = self.analyze_ml(raw_tweet)

        if ml_res.get("method") == "vader_fallback":
            return vader_res

        vader_sent = vader_res["sentiment"]
        ml_sent = ml_res["sentiment"]

        if vader_sent == ml_sent:
            ensemble_sentiment = vader_sent
            ensemble_conf = round(
                max(vader_res["confidence"], ml_res["confidence"]) * 0.95 + 0.05, 3
            )
        else:
            compound = abs(vader_res["compound_score"])
            ml_conf = ml_res["confidence"]

            if compound > 0.4 and vader_res["confidence"] >= ml_conf:
                ensemble_sentiment = vader_sent
                ensemble_conf = vader_res["confidence"]
            elif ml_conf > 0.65:
                ensemble_sentiment = ml_sent
                ensemble_conf = ml_conf
            else:
                if vader_res["confidence"] >= ml_conf:
                    ensemble_sentiment = vader_sent
                    ensemble_conf = vader_res["confidence"]
                else:
                    ensemble_sentiment = ml_sent
                    ensemble_conf = ml_conf

        return {
            "method": "ensemble",
            "sentiment": ensemble_sentiment,
            "confidence": min(1.0, ensemble_conf),
            "vader_prediction": vader_res,
            "ml_prediction": ml_res,
            "cleaned_text": vader_res["cleaned_text"],
        }

    def analyze(self, raw_tweet: str, method: str = "vader") -> Dict[str, Any]:
        metadata = self.preprocessor.extract_metadata(raw_tweet)

        if method == "ml":
            result = self.analyze_ml(raw_tweet)
        elif method == "ensemble":
            result = self.analyze_ensemble(raw_tweet)
        else:
            result = self.analyze_vader(raw_tweet)

        result["raw_text"] = raw_tweet
        result["metadata"] = metadata
        return result

    def analyze_batch(
        self,
        data: Union[List[str], pd.DataFrame],
        text_column: str = "tweet",
        method: str = "vader",
    ) -> pd.DataFrame:
        if isinstance(data, list):
            df = pd.DataFrame({text_column: data})
        else:
            df = data.copy()

        if text_column not in df.columns:
            possible_cols = [c for c in df.columns if df[c].dtype == "object"]
            if possible_cols:
                text_column = possible_cols[0]
            else:
                raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        sentiments = []
        confidences = []
        compounds = []
        mentions_list = []
        hashtags_list = []
        emojis_list = []

        for text in df[text_column].astype(str):
            res = self.analyze(text, method=method)
            sentiments.append(res["sentiment"])
            confidences.append(res["confidence"])
            compounds.append(res.get("compound_score", 0.0))
            mentions_list.append(", ".join(res["metadata"]["mentions"]))
            hashtags_list.append(", ".join(res["metadata"]["hashtags"]))
            emojis_list.append(" ".join(res["metadata"]["emojis"]))

        df["sentiment"] = sentiments
        df["confidence"] = confidences
        df["compound_score"] = compounds
        df["hashtags"] = hashtags_list
        df["mentions"] = mentions_list
        df["emojis"] = emojis_list

        return df

    def get_summary_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        if "sentiment" not in df.columns:
            raise ValueError("DataFrame must contain a 'sentiment' column.")

        total = len(df)
        counts = df["sentiment"].value_counts().to_dict()
        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}

        avg_confidence = round(float(df["confidence"].mean()), 3) if "confidence" in df.columns else 0.0
        avg_compound = round(float(df["compound_score"].mean()), 3) if "compound_score" in df.columns else 0.0

        return {
            "total_tweets": total,
            "sentiment_counts": counts,
            "sentiment_percentages": percentages,
            "average_confidence": avg_confidence,
            "average_compound": avg_compound,
        }

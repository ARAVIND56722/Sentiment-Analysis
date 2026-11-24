# Twitter Sentiment Analysis Tool

A production-ready Natural Language Processing (NLP) system designed specifically for **Twitter Data Sentiment Analysis**, classifying tweets into **Positive**, **Negative**, or **Neutral** sentiment categories.

Built using **NLTK**, **SpaCy**, and **Scikit-Learn**, this tool offers:
- Specialized Twitter preprocessing (handles mentions, hashtags, emojis, URLs, and slang contractions)
- Rule-based lexicon classification with **NLTK VADER**
- Machine learning classification with **TF-IDF + Logistic Regression** (trained on 15,000+ Twitter samples with 82.8% accuracy)
- **Ensemble Hybrid Engine** combining VADER and ML predictions
- **Interactive Streamlit Web Dashboard** with real-time analytics and CSV batch export
- **Command-Line Interface (CLI)** supporting single-tweet, batch, and interactive terminal modes
- Full **Automated Test Suite** (pytest)

---

## Architecture Overview

```
                      Raw Tweet Input
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      clean_for_vader()            clean_for_ml()
   (Preserves caps, emojis,    (Contractions, Lemmatization,
     punctuation, hashtags)        Stopwords, SpaCy POS)
               │                         │
               ▼                         ▼
          NLTK VADER                ML Pipeline
       Lexicon Analysis        (TF-IDF + Logistic Reg)
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
                     Ensemble Engine
               (High-Confidence Voting)
                            │
                            ▼
              Sentiment: POSITIVE / NEGATIVE / NEUTRAL
         Confidence Score, Metadata (Emojis, Hashtags, Mentions)
```

---

## Project Structure

```
Sentiment Analysis/
├── preprocessor.py        # Tweet text cleaning, contractions, tokenization & SpaCy pipeline
├── analyzer.py            # Core TwitterSentimentAnalyzer engine (VADER, ML, Ensemble, Batch)
├── model_trainer.py       # Trains and evaluates 3-class ML model on NLTK twitter dataset
├── cli.py                 # Rich CLI (single tweet, batch processing, interactive shell)
├── app.py                 # Streamlit interactive web dashboard
├── sample_tweets.csv      # 40+ curated benchmark tweets across domains
├── test_analyzer.py       # Comprehensive pytest unit & integration test suite
├── nltk_data/             # Local NLTK corpora (vader_lexicon, twitter_samples, etc.)
└── models/                # Trained model artifacts & evaluation metrics
    ├── sentiment_pipeline.joblib
    └── metrics.joblib
```

---

## Quickstart & Usage

### 1. Interactive Web Dashboard
Run the Streamlit app to explore tweets, batch analyze CSV files, and visualize distributions:
```bash
streamlit run app.py
```

### 2. Command-Line Interface (CLI)

#### Analyze a Single Tweet:
```bash
python cli.py --text "Loving the new update! The team did an incredible job. 🚀❤️"
```

#### Analyze with SpaCy Linguistic Token Breakdown:
```bash
python cli.py --text "Worst flight experience ever! @Airlines cancelled with no refund." --spacy
```

#### Choose Engine (`vader`, `ml`, or `ensemble`):
```bash
python cli.py --text "The weather report says 15 degrees in London." --method ensemble
```

#### Batch Analysis from CSV:
```bash
python cli.py --file sample_tweets.csv --output results.csv
```

#### Interactive Terminal Shell:
```bash
python cli.py --interactive
```

---

## Python API Usage

You can easily integrate `TwitterSentimentAnalyzer` into your own Python projects:

```python
from analyzer import TwitterSentimentAnalyzer

analyzer = TwitterSentimentAnalyzer()

# 1. Single Tweet
result = analyzer.analyze("Loving this new product! Highly recommend. 😍🔥", method="ensemble")
print(result["sentiment"])   # 'positive'
print(result["confidence"])  # 0.95
print(result["metadata"])    # {'emojis': ['😍', '🔥'], 'word_count': 7, ...}

# 2. Batch Processing
tweets = [
    "Great customer service today! 👏",
    "Flight was delayed 4 hours. Terrible. 😡",
    "The meeting will begin at 2 PM.",
]
df_results = analyzer.analyze_batch(tweets, text_column="tweet", method="vader")
print(df_results[["tweet", "sentiment", "confidence", "compound_score"]])
```

---

## Machine Learning Model Details

- **Dataset**: Built from NLTK's `twitter_samples` corpus, curated into a balanced 3-class distribution (Positive, Negative, Neutral).
- **Features**: TF-IDF Vectorizer with unigram + bigram features (`ngram_range=(1, 2)`), sublinear term frequency scaling.
- **Classifier**: Logistic Regression with balanced class weights and L2 regularization (`C=1.5`).
- **Test Set Accuracy**: **82.83%**
- **F1-Scores**:
  - Neutral: **0.99**
  - Negative: **0.76**
  - Positive: **0.74**

---

## Running the Automated Tests

Execute the comprehensive unit and integration test suite:
```bash
pytest test_analyzer.py -v
```


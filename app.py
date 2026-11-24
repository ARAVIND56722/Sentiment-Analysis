import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from analyzer import TwitterSentimentAnalyzer
from preprocessor import TweetPreprocessor

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #1DA1F2;
    }
    .sentiment-badge-pos {
        background-color: #d4edda;
        color: #155724;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 1.2rem;
        display: inline-block;
    }
    .sentiment-badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 1.2rem;
        display: inline-block;
    }
    .sentiment-badge-neu {
        background-color: #d1ecf1;
        color: #0c5460;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 1.2rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_analyzer():
    return TwitterSentimentAnalyzer()


analyzer = load_analyzer()

st.sidebar.title("Twitter Sentiment NLP")

selected_engine = st.sidebar.selectbox(
    "NLP Sentiment Engine",
    options=["vader", "ml", "ensemble"],
    format_func=lambda x: {
        "vader": "NLTK VADER",
        "ml": "ML Classifier (TF-IDF)",
        "ensemble": "Ensemble Hybrid",
    }[x],
    index=0,
)

show_linguistic = st.sidebar.checkbox("Show SpaCy Linguistic Breakdown", value=True)

nav_page = st.sidebar.radio(
    "Navigation",
    ["Live Tweet Analyzer", "Batch CSV Processing", "Multi-Model Comparison", "Model Performance"],
)

if nav_page == "Live Tweet Analyzer":
    st.header("Real-Time Tweet Sentiment Analysis")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    preset_tweet = None
    if preset_col1.button("Positive Example"):
        preset_tweet = "Loving the new update! The team did an incredible job. 🚀❤️ #excited"
    if preset_col2.button("Negative Example"):
        preset_tweet = "Flight delayed 6 hours with zero explanation. Worst customer experience ever @airlines! 😡"
    if preset_col3.button("Neutral Example"):
        preset_tweet = "The conference keynote begins tomorrow at 10 AM EST. Room 302."

    user_tweet = st.text_area(
        "Tweet Text",
        value=preset_tweet or "Loving this new update! The team did an incredible job. 🚀❤️ #awesome",
        height=100,
    )

    if st.button("Analyze Tweet", type="primary") or user_tweet:
        res = analyzer.analyze(user_tweet, method=selected_engine)
        sentiment = res["sentiment"]
        conf = res.get("confidence", 0.0)

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

        with res_col1:
            st.subheader("Classification")
            badge_class = f"sentiment-badge-{sentiment[:3]}"
            emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            st.markdown(
                f"<div class='{badge_class}'>{sentiment.upper()} {emoji_map.get(sentiment, '')}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"**Confidence:** {conf * 100:.1f}%")
            if "compound_score" in res:
                st.write(f"**Compound Score:** `{res['compound_score']:+.4f}`")

        with res_col2:
            st.subheader("Extracted Metadata")
            meta = res.get("metadata", {})
            st.markdown(f"- **Words:** {meta.get('word_count', 0)}")
            st.markdown(f"- **Characters:** {meta.get('char_count', 0)}")
            st.markdown(f"- **Hashtags:** {', '.join(meta.get('hashtags', [])) or 'None'}")
            st.markdown(f"- **Mentions:** {', '.join(meta.get('mentions', [])) or 'None'}")
            st.markdown(f"- **Emojis:** {' '.join(meta.get('emojis', [])) or 'None'}")

        with res_col3:
            st.subheader("Score Distribution")
            if "breakdown" in res:
                bd = res["breakdown"]
                chart_df = pd.DataFrame(
                    {"Sentiment": ["Positive", "Neutral", "Negative"], "Score": [bd["positive"], bd["neutral"], bd["negative"]]}
                ).set_index("Sentiment")
                st.bar_chart(chart_df)
            elif "probabilities" in res:
                probs = res["probabilities"]
                chart_df = pd.DataFrame(
                    {"Sentiment": [k.capitalize() for k in probs.keys()], "Probability": list(probs.values())}
                ).set_index("Sentiment")
                st.bar_chart(chart_df)

        if show_linguistic:
            st.markdown("---")
            st.subheader("SpaCy Token & Linguistic Structure")
            features = analyzer.preprocessor.spacy_linguistic_features(user_tweet)
            if features:
                feat_df = pd.DataFrame(features)
                st.dataframe(feat_df, use_container_width=True)

elif nav_page == "Batch CSV Processing":
    st.header("Batch Tweet Sentiment Processing")

    sample_path = "sample_tweets.csv"
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    df_to_process = None
    if uploaded_file is not None:
        df_to_process = pd.read_csv(uploaded_file)
    elif os.path.exists(sample_path):
        if st.button("Load Sample Dataset"):
            df_to_process = pd.read_csv(sample_path)
            st.session_state["batch_df"] = df_to_process

    if "batch_df" in st.session_state and df_to_process is None:
        df_to_process = st.session_state["batch_df"]

    if df_to_process is not None:
        st.write(f"Loaded **{len(df_to_process)}** rows.")
        text_cols = [c for c in df_to_process.columns if df_to_process[c].dtype == "object"]
        selected_col = st.selectbox("Select Tweet Text Column", options=text_cols, index=0)

        if st.button("Run Batch Classification", type="primary"):
            with st.spinner("Processing..."):
                enriched = analyzer.analyze_batch(df_to_process, text_column=selected_col, method=selected_engine)
                st.session_state["enriched_df"] = enriched

    if "enriched_df" in st.session_state:
        results = st.session_state["enriched_df"]
        metrics = analyzer.get_summary_metrics(results)

        st.markdown("---")
        st.subheader("Summary Metrics")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Tweets", metrics["total_tweets"])
        kpi2.metric("Positive", f"{metrics['sentiment_percentages'].get('positive', 0)}%")
        kpi3.metric("Neutral", f"{metrics['sentiment_percentages'].get('neutral', 0)}%")
        kpi4.metric("Negative", f"{metrics['sentiment_percentages'].get('negative', 0)}%")
        kpi5.metric("Avg Compound", f"{metrics['average_compound']:+.3f}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Sentiment Distribution")
            sent_counts = results["sentiment"].value_counts()
            st.bar_chart(sent_counts)

        with chart_col2:
            st.subheader("Confidence Spread")
            st.line_chart(results["confidence"].sort_values().reset_index(drop=True))

        st.subheader("Filter and Explore Results")
        filter_sent = st.multiselect(
            "Filter by Sentiment",
            options=["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
        filtered_df = results[results["sentiment"].isin(filter_sent)]
        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Analyzed CSV",
            data=csv_data,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
        )

elif nav_page == "Multi-Model Comparison":
    st.header("Multi-Model Comparison")

    test_tweet = st.text_input(
        "Compare Tweet",
        value="I do not hate the new design, but it definitely needs some improvements.",
    )

    if test_tweet:
        vader_res = analyzer.analyze(test_tweet, method="vader")
        ml_res = analyzer.analyze(test_tweet, method="ml")
        ensemble_res = analyzer.analyze(test_tweet, method="ensemble")

        comp_col1, comp_col2, comp_col3 = st.columns(3)

        with comp_col1:
            st.subheader("NLTK VADER")
            st.write(f"**Prediction:** `{vader_res['sentiment'].upper()}`")
            st.write(f"**Confidence:** `{vader_res['confidence'] * 100:.1f}%`")
            st.write(f"**Compound:** `{vader_res['compound_score']:+.4f}`")
            st.json(vader_res["breakdown"])

        with comp_col2:
            st.subheader("Machine Learning")
            st.write(f"**Prediction:** `{ml_res['sentiment'].upper()}`")
            st.write(f"**Confidence:** `{ml_res['confidence'] * 100:.1f}%`")
            if "probabilities" in ml_res:
                st.json(ml_res["probabilities"])

        with comp_col3:
            st.subheader("Ensemble Hybrid")
            st.write(f"**Decision:** `{ensemble_res['sentiment'].upper()}`")
            st.write(f"**Confidence:** `{ensemble_res['confidence'] * 100:.1f}%`")
            if vader_res["sentiment"] == ml_res["sentiment"]:
                st.success("Models agree")
            else:
                st.warning("Models disagree")

elif nav_page == "Model Performance":
    st.header("Model Evaluation & Training Metrics")
    metrics_path = "models/metrics.joblib"

    if os.path.exists(metrics_path):
        metrics = joblib.load(metrics_path)
        acc = metrics.get("accuracy", 0.0)
        report = metrics.get("classification_report", {})
        cm = metrics.get("confusion_matrix", [])

        st.metric("Test Accuracy", f"{acc * 100:.2f}%")

        st.subheader("Classification Report")
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(rep_df.style.format(precision=3), use_container_width=True)

        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["True Positive", "True Neutral", "True Negative"],
            columns=["Pred Positive", "Pred Neutral", "Pred Negative"],
        )
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Metrics not found.")

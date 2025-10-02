# app.py
# Twitter Sentiment Analysis — single-file app with training, evaluation, UI, and model persistence.
# How to run:
# 1) pip install streamlit scikit-learn pandas numpy nltk matplotlib seaborn plotly wordcloud
# 2) python -c "import nltk; [nltk.download(x) for x in ['punkt','stopwords','averaged_perceptron_tagger','vader_lexicon']]"  # one-time
# 3) streamlit run app.py

import os
import re
import io
import time
import pickle
import base64
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------
# Config & Paths
# ----------------------------
st.set_page_config(page_title="Twitter Sentiments — Single File", page_icon="🐦", layout="wide")

DATA_PATH_DEFAULT = "data/Twitter_Data.csv"  # optional; if not present, demo data will be generated
MODEL_DIR = "models"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", os.path.join(MODEL_DIR, "vectorizer.pkl"))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
def set_css():
    st.markdown(
        """
        <style>
        .result-positive {background:#e6ffed;border-left:6px solid #2e7d32;padding:10px;border-radius:6px;}
        .result-neutral {background:#f5f5f5;border-left:6px solid #616161;padding:10px;border-radius:6px;}
        .result-negative {background:#ffebee;border-left:6px solid #c62828;padding:10px;border-radius:6px;}
        .small {font-size:0.85rem;color:#666;}
        .prob-chip {display:inline-block;padding:4px 10px;border-radius:12px;margin-right:6px;background:#f0f0f0;}
        .stat-card {padding:12px;border:1px solid #eee;border-radius:8px;background:#fff;}
        </style>
        """,
        unsafe_allow_html=True,
    )

def clean_text_basic(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def transform_text(text: str, stemmer=None, stops=None) -> str:
    text = clean_text_basic(text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    if stops is None:
        stops = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stops]
    if stemmer is None:
        stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stops = set(stopwords.words("english"))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [transform_text(x, self.stemmer, self.stops) for x in X]

def feature_counts(text: str) -> dict:
    chars = len(text)
    words = len(text.split())
    sentences = max(1, len(re.split(r"[.!?]+", text)))
    return {"chars": chars, "words": words, "sentences": sentences}

def generate_wordcloud(texts, title="Word Cloud"):
    text_blob = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob if text_blob else "empty")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

def load_dataset(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Expect columns: "text" and "category" or similar
        # Normalize to Text / Category
        candidates = {c.lower(): c for c in df.columns}
        text_col = candidates.get("text") or candidates.get("tweet") or list(df.columns)[0]
        # Common label column names
        label_col = candidates.get("category") or candidates.get("sentiment") or candidates.get("target") or list(df.columns)[1]
        df = df.rename(columns={text_col: "Text", label_col: "Category"})
        # Ensure numeric mapping: -1,0,1 if required
        # If strings, map common terms
        if df["Category"].dtype == object:
            mapping = {"negative": -1, "neg": -1, "neutral": 0, "neu": 0, "positive": 1, "pos": 1}
            df["Category"] = df["Category"].str.lower().map(mapping).fillna(0).astype(int)
        return df[["Text", "Category"]]
    # Fallback small demo dataset
    data = {
        "Text": [
            "I love this phone, battery life is amazing!",
            "Worst experience ever, totally disappointed.",
            "It is okay, nothing special about it.",
            "Absolutely fantastic performance and camera.",
            "Hate the lag and frequent crashes.",
            "Service was fine, average overall.",
            "This is the best update so far!",
            "Terrible UI, confusing and slow.",
            "The product performs as expected.",
            "Great value for money, very satisfied!"
        ],
        "Category": [1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
    }
    return pd.DataFrame(data)

def train_models(X_train, y_train, random_state=42):
    preproc = TextPreprocessor()
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000)

    # Base pipelines
    pipe_mnb = Pipeline([("prep", preproc), ("tfidf", tfidf), ("clf", MultinomialNB())])
    pipe_bnb = Pipeline([("prep", preproc), ("tfidf", tfidf), ("clf", BernoulliNB())])
    pipe_lr  = Pipeline([("prep", preproc), ("tfidf", tfidf), ("clf", LogisticRegression(max_iter=200, n_jobs=None))])
    pipe_lsvc = Pipeline([("prep", preproc), ("tfidf", tfidf), ("clf", LinearSVC())])
    pipe_rf  = Pipeline([("prep", preproc), ("tfidf", tfidf), ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state))])

    # Cross-validate simple
    models = {
        "MultinomialNB": pipe_mnb,
        "BernoulliNB": pipe_bnb,
        "LogisticRegression": pipe_lr,
        "LinearSVC": pipe_lsvc,
        "RandomForest": pipe_rf
    }

    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
        cv_results.append({"model": name, "cv_accuracy_mean": scores.mean(), "cv_accuracy_std": scores.std()})

    # Fit all on full training
    for model in models.values():
        model.fit(X_train, y_train)

    # Stacking: use vectorized features; build a joint pipeline
    # We share the same preprocess + tfidf so we don't duplicate fits
    base_estimators = [
        ("mnb", MultinomialNB()),
        ("lr", LogisticRegression(max_iter=200))
    ]
    # Create a composite pipeline: preprocess -> tfidf -> stacking
    stacking = Pipeline([
        ("prep", TextPreprocessor()),
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000)),
        ("clf", StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=200),
            passthrough=False
        ))
    ])
    stacking.fit(X_train, y_train)
    models["StackingClassifier"] = stacking

    return models, pd.DataFrame(cv_results)

def evaluate_model(model, X_test, y_test, label_names={-1:"Negative",0:"Neutral",1:"Positive"}):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[-1,0,1], zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[-1,0,1])

    report = {
        "accuracy": acc,
        "precision": dict(zip(["Negative","Neutral","Positive"], pr)),
        "recall": dict(zip(["Negative","Neutral","Positive"], rc)),
        "f1": dict(zip(["Negative","Neutral","Positive"], f1)),
        "confusion_matrix": cm
    }
    return report

def save_model(model, vectorizer):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

def extract_vectorizer_from_pipeline(pipeline):
    # Given our pipelines: ("prep", TextPreprocessor), ("tfidf", TfidfVectorizer), ("clf", ...)
    try:
        return pipeline.named_steps["tfidf"]
    except Exception:
        return None

def class_to_style(c):
    if c == 1:
        return "result-positive"
    if c == 0:
        return "result-neutral"
    return "result-negative"

def class_to_label(c):
    return {1:"Positive", 0:"Neutral", -1:"Negative"}.get(int(c), "Unknown")

def probabilities_plot(proba, classes=[-1,0,1]):
    labels = [class_to_label(c) for c in classes]
    fig = go.Figure(go.Bar(x=labels, y=proba, marker_color=["#c62828", "#616161", "#2e7d32"]))
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10), yaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Sidebar
# ----------------------------
set_css()
st.sidebar.title("Twitter Sentiments")
st.sidebar.markdown("Single-file app: train, compare, and predict.")
st.sidebar.markdown("Session stats update as inferences run.")

# Data source options
data_source = st.sidebar.selectbox("Dataset source", ["Auto-detect CSV", "Upload CSV", "Demo dataset"])
st.sidebar.markdown("Expected columns: Text, Category (-1,0,1). If named differently, auto-mapping attempts will run.")
st.sidebar.markdown("Model artifacts saved to ./models")

# ----------------------------
# Load data
# ----------------------------
df = None
uploaded = None
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV with Text and Category", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        # Normalize column names
        candidates = {c.lower(): c for c in df.columns}
        text_col = candidates.get("text") or candidates.get("tweet") or list(df.columns)[0]
        label_col = candidates.get("category") or candidates.get("sentiment") or candidates.get("target") or list(df.columns)[1]
        df = df.rename(columns={text_col: "Text", label_col: "Category"})
        if df["Category"].dtype == object:
            mapping = {"negative": -1, "neg": -1, "neutral": 0, "neu": 0, "positive": 1, "pos": 1}
            df["Category"] = df["Category"].str.lower().map(mapping).fillna(0).astype(int)
else:
    if data_source == "Auto-detect CSV":
        df = load_dataset(DATA_PATH_DEFAULT)
    else:
        df = load_dataset("__no_file__")

# Derive features for EDA
df["clean"] = df["Text"].astype(str).apply(clean_text_basic)
fstats = df["Text"].astype(str).apply(feature_counts).apply(pd.Series)
df = pd.concat([df, fstats], axis=1)

# ----------------------------
# Header & Dataset preview
# ----------------------------
st.title("Twitter Sentiment Analysis 🐦📊 — Single File")
st.caption("Advanced text preprocessing, multi-model comparison, Stacking ensemble, interactive UI, and persistence — all in this one file.")

with st.expander("Dataset preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="Category", nbins=3, title="Label Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = px.histogram(df, x="words", nbins=20, title="Word Count Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Train / Load models
# ----------------------------
with st.spinner("Preparing train/test split..."):
    X_train, X_test, y_train, y_test = train_test_split(df["Text"].astype(str), df["Category"].astype(int), test_size=0.2, random_state=42, stratify=df["Category"])

# Buttons for training and loading
colA, colB, colC = st.columns([1,1,2])
with colA:
    do_train = st.button("Train Models")
with colB:
    do_load = st.button("Load Saved Best Model")

best_model_name = None
models = {}
cv_table = None
current_model = None
current_vectorizer = None

if do_load:
    m, v = load_model()
    if m is not None and v is not None:
        current_model = m
        current_vectorizer = v
        best_model_name = "Loaded Saved Model"
        st.success("Loaded saved model and vectorizer from disk.")
    else:
        st.warning("No saved model found. Train first.")

if do_train:
    with st.spinner("Training models (CV and full fit)..."):
        models, cv_table = train_models(X_train, y_train)
        cv_table = cv_table.sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
        # Choose best among simple CV models and also evaluate stacking
        # Evaluate all on validation set and pick best by accuracy
        eval_scores = []
        for name, mdl in models.items():
            y_pred = mdl.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            eval_scores.append((name, acc))
        eval_scores.sort(key=lambda x: x[1], reverse=True)
        best_model_name, best_acc = eval_scores[0]
        current_model = models[best_model_name]
        current_vectorizer = extract_vectorizer_from_pipeline(current_model)
        # Persist artifacts
        if current_vectorizer is not None:
            save_model(current_model, current_vectorizer)
        st.success(f"Best model: {best_model_name} (accuracy={best_acc:.3f}) — saved to disk.")

# ----------------------------
# Model comparison display
# ----------------------------
if cv_table is not None:
    st.subheader("Model comparison")
    st.dataframe(cv_table, use_container_width=True)
    fig_cv = px.bar(cv_table, x="model", y="cv_accuracy_mean", error_y="cv_accuracy_std", title="Cross-Validation Accuracy")
    st.plotly_chart(fig_cv, use_container_width=True)

# If a model is active (loaded or trained), evaluate and show diagnostics
if current_model is not None:
    st.subheader(f"Evaluation — {best_model_name}")
    report = evaluate_model(current_model, X_test, y_test)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{report['accuracy']:.3f}")
    c2.metric("F1 Neg", f"{report['f1']['Negative']:.3f}")
    c3.metric("F1 Neu", f"{report['f1']['Neutral']:.3f}")
    c4.metric("F1 Pos", f"{report['f1']['Positive']:.3f}")

    cm = report["confusion_matrix"]
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Pred", y="True"), x=["Neg","Neu","Pos"], y=["Neg","Neu","Pos"], title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

# ----------------------------
# Word clouds (quick EDA)
# ----------------------------
with st.expander("Word clouds by class", expanded=False):
    col1, col2, col3 = st.columns(3)
    neg_texts = df.loc[df["Category"]==-1, "clean"].values.tolist()
    neu_texts = df.loc[df["Category"]==0, "clean"].values.tolist()
    pos_texts = df.loc[df["Category"]==1, "clean"].values.tolist()
    with col1:
        st.caption("Negative")
        generate_wordcloud(neg_texts, "Negative")
    with col2:
        st.caption("Neutral")
        generate_wordcloud(neu_texts, "Neutral")
    with col3:
        st.caption("Positive")
        generate_wordcloud(pos_texts, "Positive")

# ----------------------------
# Inference UI
# ----------------------------
st.subheader("Real-time Analysis")
with st.container():
    input_text = st.text_area("Enter text", height=120, placeholder="Type or paste a tweet-like text to analyze...")
    examples = st.columns(3)
    if examples[0].button("Example Negative"):
        input_text = "This update broke everything, absolutely terrible experience."
    if examples[1].button("Example Neutral"):
        input_text = "The device was released yesterday and is available in stores."
    if examples[2].button("Example Positive"):
        input_text = "I love the new features, great performance and battery life!"

    analyze = st.button("Analyze Sentiment")

    if analyze:
        if not input_text.strip():
            st.warning("Please enter some text.")
        elif current_model is None:
            st.warning("Please train or load a model first.")
        else:
            t0 = time.time()
            # Try predict_proba; if not available (e.g., LinearSVC), fallback to decision_function
            label = current_model.predict([input_text])[0]
            proba = None
            has_proba = False
            try:
                proba = current_model.predict_proba([input_text])[0]
                # Ensure order corresponds to classes [-1,0,1] if the model has different class order
                classes = list(getattr(current_model, "classes_", [-1,0,1]))
                # Map to fixed order [-1,0,1]
                desired = [-1,0,1]
                remapped = np.zeros(3)
                for i, cls in enumerate(classes):
                    if cls in [-1,0,1]:
                        remapped[desired.index(cls)] = proba[i]
                proba = remapped
                has_proba = True
            except Exception:
                # decision_function fallback: map to pseudo probabilities via softmax
                try:
                    decision = current_model.decision_function([input_text])
                    if decision.ndim == 1:
                        # binary or one-vs-rest style; synthesize 3-class neutral in the middle
                        # map binary to [-1, 1] with neutral small prob
                        # This is a heuristic; better to pick a model with predict_proba
                        logits = np.array([decision[0], 0.0, -decision[0]])  # simple symmetric
                    else:
                        logits = decision[0]
                    ex = np.exp(logits - np.max(logits))
                    proba = ex / ex.sum()
                    has_proba = True
                except Exception:
                    has_proba = False

            latency = (time.time() - t0) * 1000.0
            lbl = class_to_label(label)
            style = class_to_style(label)

            stats = feature_counts(input_text)
            st.markdown(f'<div class="{style}"><b>Prediction:</b> {lbl}</div>', unsafe_allow_html=True)
            st.caption(f"Processed in {latency:.1f} ms • Words: {stats['words']} • Chars: {stats['chars']} • Sentences: {stats['sentences']}")

            if has_proba and proba is not None and len(proba) == 3:
                probabilities_plot(proba, classes=[-1,0,1])
                st.markdown(
                    f'<span class="prob-chip">Negative: {proba[0]:.2%}</span>'
                    f'<span class="prob-chip">Neutral: {proba[1]:.2%}</span>'
                    f'<span class="prob-chip">Positive: {proba[2]:.2%}</span>',
                    unsafe_allow_html=True
                )

# ----------------------------
# Session counters
# ----------------------------
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0
if "chars_processed" not in st.session_state:
    st.session_state.chars_processed = 0

def increment_stats(text):
    st.session_state.analysis_count += 1
    st.session_state.chars_processed += len(text)

# Tie increment to analyze button with content
if analyze and input_text.strip():
    increment_stats(input_text)

colS1, colS2, colS3 = st.columns(3)
with colS1:
    st.markdown('<div class="stat-card">Session Analyses<br><b>{}</b></div>'.format(st.session_state.analysis_count), unsafe_allow_html=True)
with colS2:
    st.markdown('<div class="stat-card">Characters Processed<br><b>{}</b></div>'.format(st.session_state.chars_processed), unsafe_allow_html=True)
with colS3:
    st.markdown('<div class="stat-card">Artifacts Path<br><span class="small">{}</span></div>'.format(MODEL_DIR), unsafe_allow_html=True)

# ----------------------------
# Download artifacts & config
# ----------------------------
st.subheader("Artifacts and Configuration")
colD1, colD2 = st.columns(2)
with colD1:
    if current_model is not None and current_vectorizer is not None and os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download model.pkl", f, file_name="model.pkl")
        with open(VECTORIZER_PATH, "rb") as f:
            st.download_button("Download vectorizer.pkl", f, file_name="vectorizer.pkl")
    else:
        st.caption("Train or load a model to enable artifact download.")

with colD2:
    st.code(
        f"""# Optional environment variables
MODEL_PATH={MODEL_PATH}
VECTORIZER_PATH={VECTORIZER_PATH}

# Streamlit configuration (run with)
# streamlit run app.py --server.port 8501 --server.address 0.0.0.0
""",
        language="bash"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built as a single-file consolidation of training, evaluation, and UI. Move to multi-file layout for production (separate training scripts, caching, CI, and tests).")

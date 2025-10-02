# app.py
# Twitter Sentiment Analysis — single-file app with training, evaluation, UI, and model persistence.
# How to run:
# 1) pip install streamlit scikit-learn pandas numpy nltk matplotlib seaborn plotly wordcloud
# 2) python -c "import nltk; [nltk.download(x) for x in ['punkt','stopwords','averaged_perceptron_tagger','vader_lexicon']]"  # one-time
# 3) streamlit run app.py

import os
import re
import io# app.py — Merged training + UI (deployment-ready)
# - Preserves your original modal.py training code and your original Streamlit UI code.
# - Adds a small sidebar control to "Train and save model" so artifacts exist on first deploy.
# - Artifacts saved to: models/vectorizer.pkl, models/model.pkl
# - Dataset expected at data/Twitter_Data.csv with columns: category, clean_text
# - If data missing, shows a helpful message.

import os
import pickle
import time
from datetime import datetime

# ----------------- Core scientific stack -----------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# ----------------- Streamlit + Plotly -----------------
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------- Ensure folders -----------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ----------------- NLTK setup -----------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        return True
    except Exception:
        return False

download_nltk_data()

# ----------------- Utility: dataset loader used by trainer -----------------
def load_training_dataframe():
    csv_path = os.path.join("data", "Twitter_Data.csv")
    if not os.path.exists(csv_path):
        return None, "Missing dataset: data/Twitter_Data.csv with columns ['category','clean_text']"
    try:
        raw = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        return None, f"Could not read CSV at {csv_path}: {e}"

    if not set(["category", "clean_text"]).issubset(set(raw.columns)):
        return None, "CSV must contain columns: 'category' and 'clean_text'"

    # Use dataset’s real columns: clean_text, category
    df = raw[['category', 'clean_text']].rename(columns={'category': 'Target', 'clean_text': 'Text'})

    # Normalize Labels: -1/0/1 to names, then encode to 0/1/2 (negative/neutral/positive)
    _target_name_map = {-1: 'negative', 0: 'neutral', 1: 'positive'}
    df['TargetName'] = df['Target'].map(_target_name_map)
    encoder = LabelEncoder()
    df['Target'] = encoder.fit_transform(df['TargetName'])  # negative→0, neutral→1, positive→2

    # Drop unused if present
    for col in ['Id', 'Date', 'Flag', 'User']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Clean nulls/dupes
    df.dropna(subset=['Text', 'Target'], inplace=True)
    df = df.drop_duplicates(keep='first')

    # Feature engineering copies of your training code
    df['num_characters'] = df['Text'].astype(str).apply(len)
    df['word_list'] = df['Text'].astype(str).apply(lambda x: nltk.word_tokenize(x))
    df['word_count'] = df['word_list'].apply(len)
    df['Sent_list'] = df['Text'].astype(str).apply(lambda x: nltk.sent_tokenize(x))
    df['Sent_count'] = df['Sent_list'].apply(len)

    # Transform function exactly as in your modal.py
    def transform_text(text):
        text = str(text).lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()
        for i in text:
            if i not in stopwords.words('english'):
                y.append(i)

        text = y[:]
        ps = PorterStemmer()
        y.clear()
        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    df['Transformed_Text'] = df['Text'].apply(transform_text)

    # Remove classes with less than 2 samples
    counts = df['Target'].value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df['Target'].isin(valid_classes)]

    return df, None

# ----------------- Trainer function (from your modal.py, consolidated) -----------------
def train_and_save_model():
    df, err = load_training_dataframe()
    if err:
        return False, err

    # CountVectorizer baseline (kept for parity, though not persisted)
    cv = CountVectorizer()
    x = cv.fit_transform(df['Transformed_Text'])
    y = df['Target'].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2, stratify=y
    )

    performance_data = []

    # GaussianNB (dense)
    try:
        gnb = GaussianNB()
        gnb.fit(x_train.toarray(), y_train)
        y_pred1 = gnb.predict(x_test.toarray())
        acc1 = accuracy_score(y_test, y_pred1)
        prec1 = precision_score(y_test, y_pred1, average='macro', zero_division=0)
        performance_data.append(("GaussianNB", acc1, prec1))
    except MemoryError:
        pass

    # MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    y_pred2 = mnb.predict(x_test)
    acc2 = accuracy_score(y_test, y_pred2)
    prec2 = precision_score(y_test, y_pred2, average='macro', zero_division=0)
    performance_data.append(("MultinomialNB", acc2, prec2))

    # BernoulliNB
    bnb = BernoulliNB()
    bnb.fit(x_train, y_train)
    y_pred3 = bnb.predict(x_test)
    acc3 = accuracy_score(y_test, y_pred3)
    prec3 = precision_score(y_test, y_pred3, average='macro', zero_division=0)
    performance_data.append(("BernoulliNB", acc3, prec3))

    # TF-IDF and models to persist
    tfidf = TfidfVectorizer()
    x_tfidf = tfidf.fit_transform(df['Transformed_Text'])
    y = df['Target'].values

    x_train, x_test, y_train, y_test = train_test_split(
        x_tfidf, y, test_size=0.2, random_state=2, stratify=y
    )

    models = {
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    best_name = None
    best_acc = -1.0
    best_model = None

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        performance_data.append((name, acc, prec))
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    # Stacking
    stack = StackingClassifier(
        estimators=[('mnb', MultinomialNB()), ('lr', LogisticRegression(solver='liblinear'))],
        final_estimator=LogisticRegression()
    )
    stack.fit(x_train, y_train)
    y_pred_stack = stack.predict(x_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    prec_stack = precision_score(y_test, y_pred_stack, average='macro', zero_division=0)
    performance_data.append(('StackingClassifier', acc_stack, prec_stack))

    # Choose final to persist: Stacking if best, else the best simple model
    final_model = stack if acc_stack >= best_acc else best_model

    # Save artifacts
    pickle.dump(tfidf, open("models/vectorizer.pkl", "wb"))
    pickle.dump(final_model, open("models/model.pkl", "wb"))

    return True, {
        "best_cv": best_name,
        "best_cv_acc": best_acc,
        "stack_acc": acc_stack,
        "performance": performance_data
    }

# ----------------- Load model & vectorizer with error handling (your UI logic) -----------------
@st.cache_resource
def load_models():
    try:
        with open("models/vectorizer.pkl", "rb") as f:
            tfidf_local = pickle.load(f)
        with open("models/model.pkl", "rb") as f:
            model_local = pickle.load(f)
        return tfidf_local, model_local, True
    except FileNotFoundError:
        return None, None, False
    except Exception:
        return None, None, False

tfidf, model, models_loaded = load_models()

# ----------------- Preprocessing for inference (your UI logic) -----------------
ps = PorterStemmer()

@st.cache_data
def get_stop_words():
    try:
        return set(stopwords.words('english'))
    except Exception:
        return set()

stop_words = get_stop_words()

def transform_text_ui(text):
    try:
        text = str(text).lower()
        tokens = nltk.word_tokenize(text)
        y = [i for i in tokens if i.isalnum()]
        y = [i for i in y if i not in stop_words]
        y = [ps.stem(i) for i in y]
        return " ".join(y)
    except Exception:
        return str(text).lower()

# ----------------- Streamlit Setup (your UI code, unchanged except training controls) -----------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Professional CSS Styling (unchanged) -----------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        :root { --primary-gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); --secondary-gradient: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%); --accent-gradient: linear-gradient(135deg, #3498db 0%, #2980b9 100%); --success-gradient: linear-gradient(135deg, #27ae60 0%, #229954 100%); --warning-gradient: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); --error-gradient: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); --glass-bg: rgba(255, 255, 255, 0.95); --glass-border: rgba(52, 73, 94, 0.1); --text-primary: #2c3e50; --text-secondary: #7f8c8d; --text-accent: #3498db; --shadow-light: 0 8px 32px rgba(44, 62, 80, 0.1); --shadow-heavy: 0 16px 48px rgba(44, 62, 80, 0.15); --bg-primary: #ecf0f1; --bg-secondary: #bdc3c7; }
        .stApp { background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%); background-attachment: fixed; font-family: 'Inter', sans-serif; }
        .stApp::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><radialGradient id="a" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%2334495e" stop-opacity="0.1"/><stop offset="100%" stop-color="%2334495e" stop-opacity="0"/></radialGradient></defs><circle cx="200" cy="200" r="100" fill="url(%23a)" opacity="0.3"><animateTransform attributeName="transform" type="translate" values="0,0;30,20;0,0" dur="15s" repeatCount="indefinite"/></circle><circle cx="800" cy="300" r="80" fill="url(%23a)" opacity="0.2"><animateTransform attributeName="transform" type="translate" values="0,0;-20,30;0,0" dur="18s" repeatCount="indefinite"/></circle><circle cx="300" cy="800" r="120" fill="url(%23a)" opacity="0.25"><animateTransform attributeName="transform" type="translate" values="0,0;25,-15;0,0" dur="12s" repeatCount="indefinite"/></circle></svg>'); pointer-events: none; z-index: 0; }
        .main-container { max-width: 900px; animation: slideUp 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94); position: relative; z-index: 1; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(50px) scale(0.95);} to { opacity: 1; transform: translateY(0) scale(1);} }
        .title-box { background: var(--glass-bg); border: 2px solid var(--glass-border); border-radius: 20px; padding: 2rem; margin-bottom: 1.5rem; box-shadow: var(--shadow-light); backdrop-filter: blur(15px); }
        .subtitle-box { background: rgba(255, 255, 255, 0.98); border: 2px solid var(--glass-border); border-radius: 16px; padding: 1.5rem; box-shadow: var(--shadow-light); backdrop-filter: blur(10px); }
        .subtitle { font-size: 1.3rem; color: var(--text-secondary); font-weight: 500; margin: 0; line-height: 1.6; }
        .input-section { animation: slideInLeft 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.5s both; }
        .input-section h3 { color: var(--text-primary); font-weight: 600; margin-bottom: 1rem; font-size: 1.4rem; }
        .stTextArea > div > div > textarea { background: rgba(255, 255, 255, 0.9) !important; border: 2px solid var(--glass-border) !important; border-radius: 16px !important; padding: 1.5rem !important; font-size: 1.1rem !important; font-family: 'Inter', sans-serif !important; transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important; backdrop-filter: blur(10px) !important; resize: vertical !important; color: var(--text-primary) !important; }
        .stTextArea > div > div > textarea:focus { border-color: #3498db !important; box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1), 0 8px 25px rgba(52, 152, 219, 0.1) !important; background: rgba(255, 255, 255, 0.98) !important; transform: scale(1.005) !important; outline: none !important; }
        .analyze-button { margin: 2rem 0; animation: slideInRight 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.7s both; }
        .stButton button { background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important; color: white !important; border-radius: 50px !important; padding: 1rem 3rem !important; font-size: 1.2rem !important; font-weight: 600 !important; font-family: 'Inter', sans-serif !important; border: none !important; cursor: pointer !important; transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important; position: relative !important; overflow: hidden !important; box-shadow: 0 6px 20px rgba(44, 62, 80, 0.3) !important; }
        .stButton button:hover { transform: translateY(-3px) scale(1.02) !important; box-shadow: 0 12px 30px rgba(44, 62, 80, 0.4) !important; }
        .stButton button:active { transform: translateY(-1px) scale(0.98) !important; }
        .loading-container { display: flex; flex-direction: column; justify-content: center; align-items: center; margin: 3rem 0; animation: fadeIn 0.5s ease-out; }
        .loading-spinner { width: 60px; height: 60px; border: 4px solid rgba(127, 140, 141, 0.2); border-top: 4px solid #34495e; border-radius: 50%; animation: spin 1s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite; margin-bottom: 1rem; }
        .loading-text { color: #7f8c8d; font-size: 1.1rem; font-weight: 500; animation: pulse 2s ease-in-out infinite; }
        @keyframes spin { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
        @keyframes pulse { 0%,100% { opacity: 0.7;} 50% { opacity: 1;} }
        .sentiment-result { animation: resultSlideIn 1s cubic-bezier(0.25, 0.46, 0.45, 0.94); margin: 3rem 0; }
        @keyframes resultSlideIn { 0% { opacity: 0; transform: scale(0.8) translateY(40px);} 50% { opacity: 0.8; transform: scale(1.05) translateY(-10px);} 100% { opacity: 1; transform: scale(1) translateY(0);} }
        .sentiment-box { padding: 3rem; border-radius: 24px; margin: 2rem 0; font-size: 2rem; text-align: center; font-weight: 700; position: relative; overflow: hidden; backdrop-filter: blur(15px); border: 2px solid var(--glass-border); transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94); }
        .sentiment-box:hover { transform: translateY(-5px) scale(1.02); }
        .positive { background: linear-gradient(135deg, rgba(39, 174, 96, 0.12), rgba(34, 153, 84, 0.08)); color: #27ae60; box-shadow: 0 12px 35px rgba(39, 174, 96, 0.2); border-color: rgba(39, 174, 96, 0.2); }
        .neutral { background: linear-gradient(135deg, rgba(243, 156, 18, 0.12), rgba(230, 126, 34, 0.08)); color: #f39c12; box-shadow: 0 12px 35px rgba(243, 156, 18, 0.2); border-color: rgba(243, 156, 18, 0.2); }
        .negative { background: linear-gradient(135deg, rgba(231, 76, 60, 0.12), rgba(192, 57, 43, 0.08)); color: #e74c3c; box-shadow: 0 12px 35px rgba(231, 76, 60, 0.2); border-color: rgba(231, 76, 60, 0.2); }
        .stats-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem; margin: 3rem 0; }
        .stat-box { background: var(--glass-bg); border-radius: 20px; padding: 2rem; text-align: center; border: 1px solid var(--glass-border); box-shadow: var(--shadow-light); transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94); backdrop-filter: blur(10px); }
        .stat-box:hover { transform: translateY(-8px); box-shadow: var(--shadow-heavy); }
        .stat-value { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
        .stat-label { color: #7f8c8d; font-size: 1rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
        .sidebar-content { background: var(--glass-bg); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; border: 1px solid var(--glass-border); backdrop-filter: blur(10px); }
        .sidebar-content h3 { color: #2c3e50; font-weight: 600; margin-bottom: 1rem; }
        .stProgress > div > div > div > div { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important; }
        .custom-footer { text-align: center; margin-top: 4rem; padding: 3rem 2rem; background: var(--glass-bg); border-radius: 20px; backdrop-filter: blur(15px); border: 1px solid var(--glass-border); animation: fadeIn 1s cubic-bezier(0.25, 0.46, 0.45, 0.94) 1s both; }
        .developer-name { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }
        .stDeployButton {display: none;} header[data-testid="stHeader"] {display: none;} .stApp > footer {display: none;} #MainMenu {display: none;} .stException {display: none;}
        @media (max-width: 768px) { .main-container { margin: 1rem; padding: 2rem;} .sentiment-box { font-size: 1.5rem; padding: 2rem;} .stats-container { grid-template-columns: 1fr;} }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(30px);} to { opacity: 1; transform: translateY(0);} }
        @keyframes slideInLeft { from { opacity: 0; transform: translateX(-40px);} to { opacity: 1; transform: translateX(0);} }
        @keyframes slideInRight { from { opacity: 0; transform: translateX(40px);} to { opacity: 1; transform: translateX(0);} }
        @keyframes expandLine { from { width: 0;} to { width: 120px;} }
        .interactive-bg { position: fixed; width: 180px; height: 180px; border-radius: 50%; pointer-events: none; z-index: -1; opacity: 0.15; filter: blur(60px); animation: float 8s ease-in-out infinite; }
        .bg-1 { background: linear-gradient(135deg, #2c3e50, #34495e); top: 15%; left: 10%; }
        .bg-2 { background: linear-gradient(135deg, #95a5a6, #7f8c8d); top: 60%; right: 15%; animation-delay: -3s; }
        .bg-3 { background: linear-gradient(135deg, #3498db, #2980b9); bottom: 15%; left: 20%; animation-delay: -6s; }
        @keyframes float { 0%, 100% { transform: translateY(0px) scale(1);} 50% { transform: translateY(-25px) scale(1.05);} }
    </style>
""", unsafe_allow_html=True)

# Interactive background elements
st.markdown("""
    <div class="interactive-bg bg-1"></div>
    <div class="interactive-bg bg-2"></div>
    <div class="interactive-bg bg-3"></div>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("### Settings")
    theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)

    st.markdown("### Model Information")
    if models_loaded:
        st.success("Models loaded successfully")
        st.info("Algorithm: Machine Learning")
        st.info("Vectorizer: TF-IDF")
        st.info("Preprocessing: NLTK")
    else:
        st.error("Models not found")
        st.warning("Please ensure model files are in the 'models' directory")

    # Training control added so first deploy can create artifacts without local commits
    st.markdown("### Training")
    if st.button("Train and save model"):
        with st.spinner("Training model on server..."):
            ok, info = train_and_save_model()
        if ok:
            st.success("Training complete. Artifacts saved to models/. Reloading...")
            # Clear caches to reload new artifacts
            load_models.clear()
            global tfidf, model, models_loaded
            tfidf, model, models_loaded = load_models()
        else:
            st.error(f"Training failed: {info}")

    st.markdown("### Session Statistics")
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'total_characters' not in st.session_state:
        st.session_state.total_characters = 0
    st.metric("Analyses Performed", st.session_state.analysis_count)
    st.metric("Total Characters Analyzed", st.session_state.total_characters)

    st.markdown("### Quick Examples")
    examples = {
        "Positive": "I absolutely love this product! It exceeded all my expectations and the customer service was amazing.",
        "Negative": "This was the worst experience ever. The product broke immediately and customer service was terrible.",
        "Neutral": "The product is okay. It works as described but nothing special about it."
    }
    for label, text in examples.items():
        if st.button(label, key=f"example_{label}"):
            st.session_state.example_text = text

    st.markdown("### How It Works")
    with st.expander("View Process"):
        st.markdown("""
        1. Text Preprocessing: Removes noise and normalizes text
        2. Vectorization: Converts text to numerical features using TF-IDF
        3. Prediction: Uses trained ML model to classify sentiment
        4. Confidence: Shows probability scores for each class
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Main Application -----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <div class="title-box">
            <h1 class="main-title">AI Sentiment Analyzer Pro</h1>
        </div>
        <div class="subtitle-box">
            <p class="subtitle">Advanced emotional intelligence powered by machine learning</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# If artifacts missing and user didn’t click train, stop with same messaging as your UI
if not models_loaded:
    st.error("Model files not found!")
    st.markdown("""
    Required Files:
    - models/vectorizer.pkl
    - models/model.pkl

    Use the sidebar button "Train and save model" to generate artifacts on the server,
    or commit the files to the repository under models/.
    """)
    st.stop()

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("### Enter Your Text")

default_text = ""
if 'example_text' in st.session_state:
    default_text = st.session_state.example_text
    del st.session_state.example_text

user_input = st.text_area(
    "",
    value=default_text,
    height=150,
    placeholder="Type or paste your text here for sentiment analysis...\n\nExamples:\n• Product reviews\n• Social media posts\n• Customer feedback\n• Survey responses",
    help="Tip: The more descriptive your text, the more accurate the sentiment analysis will be."
)

char_count = len(user_input)
word_count = len(user_input.split()) if user_input.strip() else 0

c1, c2, c3 = st.columns(3)
with c1: st.metric("Characters", char_count)
with c2: st.metric("Words", word_count)
with c3: st.metric("Reading Time", f"{max(1, word_count // 200)} min")

st.markdown('</div>', unsafe_allow_html=True)

# Analysis Button
st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
if st.button("Analyze Sentiment", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif char_count < 5:
        st.warning("Please enter at least 5 characters for accurate analysis.")
    else:
        st.session_state.analysis_count += 1
        st.session_state.total_characters += char_count

        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Analyzing sentiment...</div>
            </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        loading_placeholder.empty()
        progress_bar.empty()

        try:
            transformed_text = transform_text_ui(user_input)
            vector_input = tfidf.transform([transformed_text]).toarray()
            prediction = model.predict(vector_input)[0]
            prediction_proba = None
            try:
                prediction_proba = model.predict_proba(vector_input)[0]
            except AttributeError:
                st.info("Probability scores not available for this model.")

            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment_label = sentiment_map.get(int(prediction), str(prediction))
            sentiment_text = {'positive':'Positive','neutral':'Neutral','negative':'Negative'}.get(sentiment_label, 'Unknown')

            st.markdown('<div class="sentiment-result">', unsafe_allow_html=True)
            st.markdown(f'''
                <div class="sentiment-box {sentiment_label}">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">Sentiment: {sentiment_text}</div>
                    <div style="font-size: 1.2rem; font-weight: 400; opacity: 0.8;">
                        {datetime.now().strftime("Analyzed on %B %d, %Y at %I:%M %p")}
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            if prediction_proba is not None:
                confidence = float(max(prediction_proba) * 100.0)

                st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.markdown(f'''
                        <div class="stat-box">
                            <div class="stat-value">{confidence:.1f}%</div>
                            <div class="stat-label">Confidence</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with sc2:
                    st.markdown(f'''
                        <div class="stat-box">
                            <div class="stat-value">{word_count}</div>
                            <div class="stat-label">Words</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with sc3:
                    processed_words = len(transformed_text.split()) if transformed_text else 0
                    st.markdown(f'''
                        <div class="stat-box">
                            <div class="stat-value">{processed_words}</div>
                            <div class="stat-label">Processed</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with sc4:
                    st.markdown(f'''
                        <div class="stat-box">
                            <div class="stat-value">{'< 1s'}</div>
                            <div class="stat-label">Time</div>
                        </div>
                    ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### Detailed Analysis")
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Neutral', 'Positive'],
                    'Probability': prediction_proba * 100.0
                })
                fig = px.bar(
                    prob_df, x='Sentiment', y='Probability',
                    color='Probability',
                    color_continuous_scale=['#e74c3c', '#f39c12', '#27ae60'],
                    title="Sentiment Probability Distribution"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", size=14, color='#2c3e50'),
                    title_font_size=20, title_font_color='#2c3e50',
                    showlegend=False, height=400
                )
                fig.update_traces(
                    texttemplate='%{y:.1f}%', textposition='outside',
                    marker_line_color='rgba(44, 62, 80, 0.2)', marker_line_width=2
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Detailed Breakdown"):
                    for sentiment, prob in zip(['Negative','Neutral','Positive'], prediction_proba * 100.0):
                        st.markdown(f"{sentiment}: {prob:.2f}%")
                        st.progress(min(

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

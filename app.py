# app.py — Merged training + UI (deployment-ready; fixed global declaration error)
# Saves artifacts to models/vectorizer.pkl and models/model.pkl
# Expects data/Twitter_Data.csv with columns: category, clean_text

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

    # Feature engineering
    df['num_characters'] = df['Text'].astype(str).apply(len)
    df['word_list'] = df['Text'].astype(str).apply(lambda x: nltk.word_tokenize(x))
    df['word_count'] = df['word_list'].apply(len)
    df['Sent_list'] = df['Text'].astype(str).apply(lambda x: nltk.sent_tokenize(x))
    df['Sent_count'] = df['Sent_list'].apply(len)

    # Transform function (training)
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

# ----------------- Trainer function (from modal.py, consolidated) -----------------
def train_and_save_model():
    df, err = load_training_dataframe()
    if err:
        return False, err

    # CountVectorizer baseline (not persisted)
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

    # Choose final to persist
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

# ----------------- Load model & vectorizer with error handling (UI logic) -----------------
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

# First load into session_state so no global rebinds are needed
if "tfidf" not in st.session_state or "model" not in st.session_state or "models_loaded" not in st.session_state:
    _tfidf, _model, _loaded = load_models()
    st.session_state.tfidf = _tfidf
    st.session_state.model = _model
    st.session_state.models_loaded = _loaded

tfidf = st.session_state.tfidf
model = st.session_state.model
models_loaded = st.session_state.models_loaded

# ----------------- Preprocessing for inference (UI logic) -----------------
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

# ----------------- Streamlit Setup -----------------
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Professional CSS Styling -----------------
st.markdown(""" <style> /* full CSS omitted for brevity in this message, keep your previous CSS block unchanged here */ </style> """, unsafe_allow_html=True)

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

    # Training control so first deploy can create artifacts (uses session_state instead of global)
    st.markdown("### Training")
    if st.button("Train and save model"):
        with st.spinner("Training model on server..."):
            ok, info = train_and_save_model()
        if ok:
            st.success("Training complete. Artifacts saved to models/. Reloading...")
            load_models.clear()
            _tfidf, _model, _loaded = load_models()
            st.session_state.tfidf = _tfidf
            st.session_state.model = _model
            st.session_state.models_loaded = _loaded
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

# Refresh local references after possible training
tfidf = st.session_state.tfidf
model = st.session_state.model
models_loaded = st.session_state.models_loaded

# If artifacts missing and user didn’t click train, stop with same messaging
if not models_loaded or tfidf is None or model is None:
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
                        st.progress(min(max(prob / 100.0, 0.0), 1.0))

                st.markdown("### Interpretation")
                if confidence >= 90:
                    confidence_level = "Very High"; confidence_color = "#27ae60"
                elif confidence >= 75:
                    confidence_level = "High"; confidence_color = "#3498db"
                elif confidence >= 60:
                    confidence_level = "Moderate"; confidence_color = "#f39c12"
                else:
                    confidence_level = "Low"; confidence_color = "#e74c3c"

                interpretation = {
                    'positive': "The text expresses positive emotions, satisfaction, or favorable opinions.",
                    'neutral': "The text is factual, objective, or lacks strong emotional indicators.",
                    'negative': "The text expresses negative emotions, dissatisfaction, or unfavorable opinions."
                }.get(sentiment_label, "Unable to determine sentiment clearly.")

                certainty_text = ("The model is very confident in this prediction."
                                  if confidence >= 80 else
                                  "The model has moderate confidence in this prediction."
                                  if confidence >= 60 else
                                  "The model has low confidence in this prediction. Consider providing more context.")

                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.1); 
                    border-radius: 12px; 
                    padding: 1.5rem; 
                    margin: 1rem 0;
                    border-left: 4px solid {confidence_color};
                    backdrop-filter: blur(10px);
                ">
                    <strong>Analysis:</strong> {interpretation}<br><br>
                    <strong>Confidence Level:</strong> <span style="color: {confidence_color};">{confidence_level} ({confidence:.1f}%)</span><br>
                    <strong>Model Certainty:</strong> {certainty_text}
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Please try again with different text or check your model files.")

st.markdown('</div>', unsafe_allow_html=True)

# Professional Footer
st.markdown("""
    <div class="custom-footer">
        <div style="font-size: 1.3rem; margin-bottom: 1rem;">
            Crafted with precision by <span class="developer-name">Shravan Chafekar</span>
        </div>
        <div style="font-size: 1rem; color: var(--text-secondary); margin-bottom: 1.5rem;">
            Powered by Python • Scikit-learn • Streamlit • Plotly
        </div>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <span>Advanced AI</span>
            <span>Accurate Predictions</span>
            <span>Real-time Analysis</span>
            <span>Responsive Design</span>
        </div>
        <div style="font-size: 0.9rem; color: var(--text-secondary); opacity: 0.7;">
            © 2025 AI Sentiment Analyzer Pro. Built with expertise for better understanding.
        </div>
    </div>
""", unsafe_allow_html=True)

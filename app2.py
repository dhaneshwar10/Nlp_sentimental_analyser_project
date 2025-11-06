import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/best_sentiment_model.pkl")

model = load_model()

# --- Header ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>🧠 Sentiment Analyzer</h1>
    <p style='text-align: center; font-size: 18px;'>Type a review below and let the model predict its sentiment.</p>
    """,
    unsafe_allow_html=True
)

# --- Input box ---
with st.form("sentiment_form"):
    text = st.text_area("✍️ Enter your review here:", height=150, placeholder="E.g. I absolutely loved this movie!")
    submitted = st.form_submit_button("🔍 Analyze")

# Prediction 
if submitted:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        proba = model.predict_proba([text])[0]
        pred = int(proba.argmax())
        label = "Positive 😊" if pred == 1 else "Negative 😞"
        confidence = proba.max()

        #  Result card 
        st.markdown(
            f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
                <h2 style='color: {"green" if pred == 1 else "red"};'>Prediction: {label}</h2>
                <p style='font-size: 18px;'>Confidence: <strong>{confidence:.2f}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

#  Footer 
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px; color: gray;'>
        Built with ❤️ using Streamlit & scikit-learn | Dhanesh's NLP Project
    </p>
    """,
    unsafe_allow_html=True
)
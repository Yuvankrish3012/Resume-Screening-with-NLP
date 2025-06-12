import streamlit as st
import pickle
import re
import numpy as np

# === Absolute Paths to Pickle Files ===
model_path = r"D:\ML PROJECTS\Resume Screening with NLP\resume_model.pkl"
vectorizer_path = r"D:\ML PROJECTS\Resume Screening with NLP\resume_vectorizer.pkl"
label_encoder_path = r"D:\ML PROJECTS\Resume Screening with NLP\resume_label_encoder.pkl"

# === Load Model, Vectorizer, Label Encoder ===
model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))
label_encoder = pickle.load(open(label_encoder_path, "rb"))

# === Clean Text Function ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# === Streamlit App ===
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>📄 Resume Role Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Paste resume content below to detect its job role category</p>", unsafe_allow_html=True)

# === Sidebar with Professional Icon ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942748.png", width=120)
    st.title("Resume Classifier")
    st.markdown("🧠 NLP-based job role prediction")
    st.markdown("---")
    st.write("👤 Created by **Yuvan Krishnan**")
    st.markdown("📍 `TF-IDF + Logistic Regression`")
    st.markdown("🌐 Streamlit-based GUI")

# === User Input ===
resume_text = st.text_area("📋 Paste Resume Text", placeholder="Paste full resume text here...", height=300)

# === Predict Button ===
if st.button("🔍 Predict Role", use_container_width=True):
    if not resume_text.strip():
        st.warning("Please enter or paste resume content.")
    else:
        cleaned = clean_text(resume_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        # Confidence scores
        probas = model.predict_proba(vectorized)[0]
        top_3 = np.argsort(probas)[::-1][:3]
        top_labels = label_encoder.inverse_transform(top_3)
        top_scores = probas[top_3]

        # Show results
        st.success(f"✅ Predicted Job Category: **{predicted_label}**")
        st.markdown("---")
        st.markdown("#### 🔢 Top 3 Predictions:")
        for i in range(3):
            st.markdown(f"- **{top_labels[i]}** — `{top_scores[i]*100:.2f}%` confidence")

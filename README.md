# Resume-Screening-with-NLP

This project uses Natural Language Processing (NLP) and Machine Learning to classify resumes into predefined job categories like `Data Science`, `Web Designing`, `HR`, etc., by analyzing the text content of the resumes.

---

## 📂 Dataset

- **Name**: [Resume Dataset – Gaurav Dutta](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- **Format**: CSV (`UpdatedResumeDataSet.csv`)
- **Fields**:
  - `Resume`: Raw resume text
  - `Category`: Job label (target)

---

## 🔍 Problem Statement

Build a smart system to **automatically classify resumes** into job roles using:
- ✅ Text preprocessing
- ✅ TF-IDF vectorization
- ✅ Logistic Regression classifier
- ✅ Streamlit-based GUI

---

## 📊 Model Performance

| Metric       | Score |
|--------------|-------|
| ✅ Accuracy   | 0.99  |
| ✅ Precision  | 1.00  |
| ✅ Recall     | 0.99  |
| ✅ F1 Score   | 0.99  |

📋 **Classification Report**  
<sub>(Click to expand)</sub>

<PASTE CLASSIFICATION REPORT HERE> ```
📉 Confusion Matrix

![image](https://github.com/user-attachments/assets/33b1ebc0-e6d7-4c6a-88f1-053e974a5f74)

🌥️ Word Clouds by Category
Below are the word clouds that show the most frequent keywords in resumes across different categories.

![image](https://github.com/user-attachments/assets/af368311-1601-4cf9-ba42-cd6f525c597e)


🧪 Tech Stack
Python

Pandas, scikit-learn

TF-IDF for feature extraction

Logistic Regression

Matplotlib, Seaborn, WordCloud

Streamlit for UI

📦 Files and Folders
csharp
Copy
Edit
📁 Resume Screening with NLP/
│
├── UpdatedResumeDataSet.csv        # Dataset
├── resume_model.pkl                # Trained model
├── resume_vectorizer.pkl           # TF-IDF vectorizer
├── resume_label_encoder.pkl        # Label encoder
├── resume_app.py                   # Streamlit frontend
└── README.md
🚀 How to Run the App
Make sure you're in the correct environment (myenv) and directory.

bash
Copy
Edit
streamlit run "D:\ML PROJECTS\Resume Screening with NLP\resume_app.py"
📸 Streamlit App Preview

![image](https://github.com/user-attachments/assets/bab24862-1d64-4c38-8b8f-8d602d047fc6)


✅ Features Implemented
 Resume cleaning & preprocessing

 Resume role classification

 TF-IDF + Logistic Regression pipeline

 Word cloud visualizations

 Confusion matrix & classification report

 Streamlit-based UI

 Top 3 role predictions with confidence

🔮 Future Improvements (Optional Enhancements)
 Add resume file upload (.pdf or .txt)

 Use BERT or LSTM for deeper role matching

 Deploy to Streamlit Cloud with public share link

 Display recommended job descriptions

👨‍💻 Built by
Yuvan Krishnan
📌 B.Tech @ SRM Institute of Science and Technology
🔗 GitHub Profile

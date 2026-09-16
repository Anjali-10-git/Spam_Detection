# Spam_Detection
A simple Spam Detection Web Application built with Python, Scikit-learn, and Streamlit. The application analyzes email/SMS text and predicts whether the message is Spam or Not Spam (Ham).

Technologies Used
Python
Streamlit — Web application interface
Scikit-learn — Machine learning
Joblib / Pickle — Model loading
Regular Expressions (re) — Text preprocessing

| File                 | Description                    |
| -------------------- | ------------------------------ |
| `app.py`             | Main Streamlit application     |
| `Spam_Detection.pkl` | Trained machine-learning model |
| `requirements.txt`   | Required Python libraries      |
| `README.md`          | Project documentation          |

How It Works-

User enters message
        ↓
Text preprocessing
        ↓
Machine-learning model
        ↓
Spam signal analysis
        ↓
Final classification
        ↓
Spam / Not Spam

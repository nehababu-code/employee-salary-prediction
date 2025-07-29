import streamlit as st
import pandas as pd
import os
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

# Load image (optional fallback)
def load_banner():
    banner_path = "banner.jpg"  # Ensure this file exists in the same folder
    if os.path.exists(banner_path):
        st.image(banner_path, use_column_width=True)
    else:
        st.warning("👋 Upload a file named `banner.jpg` in the app folder to display the welcome image.")

# Main homepage
def main():
    st.markdown("## 💼 Welcome to Employee Salary Predictor")

    load_banner()

    st.markdown("""
    This app predicts whether an employee earns **more than $50K/year** or not based on their profile.
    
    👉 Click the button below to continue to the prediction form.
    """)

    if st.button("🔍 Predict Salary"):
        st.session_state.page = "predict"

# Prediction page
def prediction_page():
    st.markdown("## 🧠 Salary Prediction")

    # Load data for encoding consistency
    data = pd.read_csv("adult 3.csv")

    # Drop rows with missing values
    data = data.dropna()

    # Split into features and target
    X = data.drop('class', axis=1)
    y = data['class']

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Train a model (Random Forest for demo)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)

    # Input from user
    def user_input():
        age = st.slider('Age', 18, 90, 30)
        workclass = st.selectbox('Workclass', label_encoders['workclass'].classes_)
        education = st.selectbox('Education', label_encoders['education'].classes_)
        occupation = st.selectbox('Occupation', label_encoders['occupation'].classes_)
        sex = st.selectbox('Sex', label_encoders['sex'].classes_)

        input_dict = {
            'age': age,
            'workclass': label_encoders['workclass'].transform([workclass])[0],
            'education': label_encoders['education'].transform([education])[0],
            'occupation': label_encoders['occupation'].transform([occupation])[0],
            'sex': label_encoders['sex'].transform([sex])[0]
        }

        return pd.DataFrame([input_dict])

    input_df = user_input()

    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    result = "💰 Earns >50K/year" if prediction == ">50K" else "📉 Earns ≤50K/year"
    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {prediction_proba[prediction == model.classes_][0]*100:.2f}%")

# Handle navigation
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main()
else:
    prediction_page()

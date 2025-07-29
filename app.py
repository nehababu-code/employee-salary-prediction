import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Streamlit page configuration ---
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# --- Hide Streamlit footer and hamburger menu ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Page Routing using session state ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- HOME PAGE ---
if st.session_state.page == 'home':
    st.title("💼 Welcome to Employee Salary Predictor")

    # Display banner if available
    if os.path.exists("banner.jpg"):
        st.image("banner.jpg", use_column_width=True)
    else:
        st.warning("👋 Upload a file named **banner.jpg** in the app folder to display the welcome image.")

    st.markdown("This app predicts whether an employee earns more than $50K/year based on user inputs.")

    if st.button("🔮 Predict Salary"):
        st.session_state.page = 'predict'

# --- PREDICTION PAGE ---
elif st.session_state.page == 'predict':
    st.title("🔍 Salary Prediction")

    try:
        data = pd.read_csv("adult 3.csv")
    except FileNotFoundError:
        st.error("Dataset 'adult 3.csv' not found. Please upload it in the app folder.")
        st.stop()

    # Clean and preprocess
    if 'educational-num' in data.columns:
        data.drop('educational-num', axis=1, inplace=True)

    data.replace(' ?', pd.NA, inplace=True)
    data.dropna(inplace=True)

    # Capture original values for dropdowns
    original_values = {}
    for col in data.select_dtypes(include='object').columns:
        original_values[col] = sorted([val for val in data[col].unique() if val.strip() != '?'])

    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop('income', axis=1)
    y = data['income']
    model = RandomForestClassifier()
    model.fit(X, y)

    # --- User input form ---
    st.subheader("Enter Employee Details")

    def get_user_input():
        age = st.slider("Age", 18, 70, 30)
        workclass = st.selectbox("Workclass", original_values['workclass'])
        education = st.selectbox("Education", original_values['education'])
        marital = st.selectbox("Marital Status", original_values['marital-status'])
        occupation = st.selectbox("Occupation", original_values['occupation'])
        relationship = st.selectbox("Relationship", original_values['relationship'])
        race = st.selectbox("Race", original_values['race'])
        gender = st.selectbox("Gender", original_values['gender'])
        hours = st.slider("Hours per week", 1, 99, 40)
        country = st.selectbox("Native Country", original_values['native-country'])

        return pd.DataFrame([{
            'age': age,
            'workclass': label_encoders['workclass'].transform([workclass])[0],
            'fnlwgt': 200000,
            'education': label_encoders['education'].transform([education])[0],
            'marital-status': label_encoders['marital-status'].transform([marital])[0],
            'occupation': label_encoders['occupation'].transform([occupation])[0],
            'relationship': label_encoders['relationship'].transform([relationship])[0],
            'race': label_encoders['race'].transform([race])[0],
            'gender': label_encoders['gender'].transform([gender])[0],
            'capital-gain': 0,
            'capital-loss': 0,
            'hours-per-week': hours,
            'native-country': label_encoders['native-country'].transform([country])[0],
        }])

    input_df = get_user_input()

    if st.button("📊 Predict"):
        prediction = model.predict(input_df)[0]
        result = label_encoders['income'].inverse_transform([prediction])[0]
        st.success(f"💰 Predicted Income: **{result}**")

    if st.button("⬅️ Back to Home"):
        st.session_state.page = 'home'

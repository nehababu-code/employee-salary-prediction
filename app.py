import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("adult 3.csv")

# Encode categorical features
@st.cache_data
def preprocess_data(data):
    label_encoders = {}
    for column in data.select_dtypes(include='object'):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

# Predict salary function
def predict(model, label_encoders, input_df):
    input_encoded = input_df.copy()
    for col in input_encoded.columns:
        if col in label_encoders:
            le = label_encoders[col]
            input_encoded[col] = le.transform([input_encoded[col][0]])
    return model.predict(input_encoded)[0]

# Page 1: Home
def home():
    st.title("💼 Welcome to Employee Salary Predictor")
    if os.path.exists("banner.jpg"):
        st.image("banner.jpg", use_column_width=True)
    else:
        st.warning("👋 Upload a file named **banner.jpg** in the app folder to display the welcome image.")
    
    st.markdown("This app predicts whether an employee earns more than $50K/year based on demographic features.")
    
    if st.button("🔮 Predict Salary"):
        st.session_state.page = "predict"

# Page 2: Prediction
def prediction_page():
    st.title("🔮 Salary Prediction")

    # Load and process
    data = load_data()
    processed_data, label_encoders = preprocess_data(data)
    X = processed_data.drop('income', axis=1)
    y = processed_data['income']
    model = RandomForestClassifier()
    model.fit(X, y)

    # Input UI
    def user_input():
        age = st.slider('Age', 18, 90, 30)
        workclass = st.selectbox('Workclass', label_encoders['workclass'].classes_)
        education = st.selectbox('Education', label_encoders['education'].classes_)
        marital = st.selectbox('Marital Status', label_encoders['marital-status'].classes_)
        occupation = st.selectbox('Occupation', label_encoders['occupation'].classes_)
        relationship = st.selectbox('Relationship', label_encoders['relationship'].classes_)
        race = st.selectbox('Race', label_encoders['race'].classes_)
        gender = st.selectbox('Gender', label_encoders['gender'].classes_)
        hours = st.slider('Hours per week', 1, 99, 40)
        country = st.selectbox('Native Country', label_encoders['native-country'].classes_)
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 100000, 0)

        return pd.DataFrame({
            'age': [age],
            'workclass': [workclass],
            'fnlwgt': [200000],  # Placeholder
            'education': [education],
            'educational-num': [10],  # Placeholder
            'marital-status': [marital],
            'occupation': [occupation],
            'relationship': [relationship],
            'race': [race],
            'gender': [gender],
            'capital-gain': [capital_gain],
            'capital-loss': [capital_loss],
            'hours-per-week': [hours],
            'native-country': [country]
        })

    input_df = user_input()

    if st.button("🎯 Predict"):
        result = predict(model, label_encoders, input_df)
        label = label_encoders['income'].inverse_transform([result])[0]
        st.success(f"💰 Predicted Salary Category: **{label}**")

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"

# App Router
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home()
elif st.session_state.page == "predict":
    prediction_page()

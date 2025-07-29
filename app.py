import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    return df

data = load_data()

# Show banner image
st.title("💼 Welcome to Employee Salary Predictor")
try:
    st.image("banner.jpg", use_container_width=True)
except:
    st.warning("👋 Upload a file named banner.jpg in the app folder to display the welcome image.")

# Button to go to prediction page
if st.button("👉 Predict Salary"):
    st.session_state.page = "predict"

# Switch to prediction page if user clicked button
if "page" in st.session_state and st.session_state.page == "predict":

    st.header("📊 Salary Prediction Form")

    # Separate features and target
    X = data.drop('income', axis=1)
    y = data['income']

    # Encode categorical variables
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Collect user input
    def user_input():
        user_data = {}
        for column in X.columns:
            if column in label_encoders:
                options = list(label_encoders[column].classes_)
                user_data[column] = st.selectbox(column.capitalize(), options)
            else:
                user_data[column] = st.number_input(column.capitalize(), value=float(data[column].mean()))
        return pd.DataFrame([user_data])

    input_df = user_input()

    # Encode input
    for column in label_encoders:
        input_df[column] = label_encoders[column].transform(input_df[column])

    # Make prediction
    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)
        result = le_target.inverse_transform(prediction)[0]
        st.success(f"💰 Predicted Salary: {result}")

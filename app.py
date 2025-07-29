import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page state handling
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# === WELCOME PAGE ===
if st.session_state.page == 'welcome':
    st.title("💼 Employee Salary Predictor")

    try:
        st.image("banner.jpg", use_container_width=True)
    except:
        st.warning("👋 Upload a file named `banner.jpg` in the app folder to display the welcome image.")

    st.markdown("Welcome! This app predicts whether an employee earns more than $50K/year based on their profile.")

    if st.button("🚀 Predict Salary"):
        st.session_state.page = 'predict'
        st.experimental_rerun()

# === PREDICTION PAGE ===
elif st.session_state.page == 'predict':
    st.title("📊 Predict Employee Income")

    # Load dataset
    data = pd.read_csv("adult 3.csv")

    # Drop unnecessary column if it exists
    if 'educational-num' in data.columns:
        data.drop('educational-num', axis=1, inplace=True)

    # Replace missing values and drop rows
    data.replace(' ?', pd.NA, inplace=True)
    data.dropna(inplace=True)

    # Store original values for dropdowns
    original_values = {}
    for col in data.select_dtypes(include='object').columns:
        clean_vals = [val for val in data[col].unique() if val.strip() != '?']
        original_values[col] = sorted(clean_vals)

    # Label encode categorical columns
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Train model
    X = data.drop('income', axis=1)
    y = data['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Input form
    def user_input():
        age = st.number_input("Age", 18, 100, 30)
        workclass = st.selectbox("Workclass", original_values['workclass'])
        education = st.selectbox("Education", original_values['education'])
        marital_status = st.selectbox("Marital Status", original_values['marital-status'])
        occupation = st.selectbox("Occupation", original_values['occupation'])
        relationship = st.selectbox("Relationship", original_values['relationship'])
        race = st.selectbox("Race", original_values['race'])
        gender = st.selectbox("Gender", original_values['gender'])
        hours_per_week = st.slider("Hours per week", 1, 99, 40)
        native_country = st.selectbox("Native Country", original_values['native-country'])

        input_data = {
            'age': age,
            'workclass': label_encoders['workclass'].transform([workclass])[0],
            'fnlwgt': 200000,
            'education': label_encoders['education'].transform([education])[0],
            'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
            'occupation': label_encoders['occupation'].transform([occupation])[0],
            'relationship': label_encoders['relationship'].transform([relationship])[0],
            'race': label_encoders['race'].transform([race])[0],
            'gender': label_encoders['gender'].transform([gender])[0],
            'capital-gain': 0,
            'capital-loss': 0,
            'hours-per-week': hours_per_week,
            'native-country': label_encoders['native-country'].transform([native_country])[0],
        }

        return pd.DataFrame([input_data])

    input_df = user_input()

    if st.button("Predict Income"):
        prediction = model.predict(input_df)[0]
        result = label_encoders['income'].inverse_transform([prediction])[0]
        st.success(f"🧾 Predicted Income: {result}")

    if st.button("🔙 Back to Home"):
        st.session_state.page = 'welcome'
        st.experimental_rerun()

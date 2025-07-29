import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Load image (optional)
if os.path.exists("banner.jpg"):
    st.image("banner.jpg", use_container_width=True)
else:
    st.warning("👋 Upload a file named **banner.jpg** in the app folder to display the welcome image.")

# Navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "🔍 Predict Salary"])

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")

    if 'educational-num' in df.columns:
        df.drop('educational-num', axis=1, inplace=True)

    df.replace(' ?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Store original values for dropdowns
    dropdown_values = {
        col: sorted([val for val in df[col].unique() if str(val).strip() != '?'])
        for col in df.select_dtypes(include='object').columns
    }

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders, dropdown_values

# Prediction logic
def predict_salary(model, label_encoders, dropdowns):
    st.subheader("🔍 Fill in the details")

    age = st.number_input("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", dropdowns['workclass'])
    education = st.selectbox("Education", dropdowns['education'])
    marital = st.selectbox("Marital Status", dropdowns['marital-status'])
    occupation = st.selectbox("Occupation", dropdowns['occupation'])
    relationship = st.selectbox("Relationship", dropdowns['relationship'])
    race = st.selectbox("Race", dropdowns['race'])
    gender = st.selectbox("Gender", dropdowns['gender'])
    hours = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", dropdowns['native-country'])

    # Hardcoded for now
    fnlwgt = 200000
    capital_gain = 0
    capital_loss = 0

    input_dict = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'fnlwgt': fnlwgt,
        'education': label_encoders['education'].transform([education])[0],
        'marital-status': label_encoders['marital-status'].transform([marital])[0],
        'occupation': label_encoders['occupation'].transform([occupation])[0],
        'relationship': label_encoders['relationship'].transform([relationship])[0],
        'race': label_encoders['race'].transform([race])[0],
        'gender': label_encoders['gender'].transform([gender])[0],
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours,
        'native-country': label_encoders['native-country'].transform([native_country])[0],
    }

    df_input = pd.DataFrame([input_dict])

    if st.button("Predict Income"):
        pred = model.predict(df_input)[0]
        result = label_encoders['income'].inverse_transform([pred])[0]
        st.success(f"🧾 Predicted Income: **{result}**")

# MAIN
def main():
    try:
        df, encoders, dropdowns = load_data()

        X = df.drop('income', axis=1)
        y = df['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        if page == "🏠 Home":
            st.title("💼 Welcome to Employee Salary Predictor")
            st.markdown("This app predicts whether an employee earns more than $50K/year.")
            st.info("Go to the **'Predict Salary'** tab in the sidebar to begin.")

        elif page == "🔍 Predict Salary":
            predict_salary(model, encoders, dropdowns)

    except FileNotFoundError:
        st.error("❌ File `adult 3.csv` not found. Please upload it in the root folder.")

if __name__ == "__main__":
    main()

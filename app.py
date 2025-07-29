import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

# Function to load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("adult 3.csv")

    if 'educational-num' in data.columns:
        data.drop('educational-num', axis=1, inplace=True)

    data.replace(' ?', pd.NA, inplace=True)
    data.dropna(inplace=True)

    original_values = {}
    for col in data.select_dtypes(include='object').columns:
        clean_vals = [val for val in data[col].unique() if val.strip() != '?']
        original_values[col] = sorted(clean_vals)

    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, original_values, label_encoders

# Load and prepare data
data, original_values, label_encoders = load_data()

# Train model
X = data.drop('income', axis=1)
y = data['income']
model = RandomForestClassifier()
model.fit(X, y)

# Routing: use query parameters
query_params = st.query_params
page = query_params.get("page", "home")

# Page 1: Home page with banner and button
if page == "home":
    st.title("💼 Welcome to the Employee Salary Predictor")

    try:
        st.image("banner.jpg", use_container_width=True)
    except:
        st.warning("👋 Upload a file named banner.jpg in the app folder to show the welcome image.")

    st.markdown("Click the button below to predict employee salary based on details:")

    if st.button("🔮 Predict Salary"):
        st.switch_page("?page=predict")

# Page 2: Prediction form
elif page == "predict":
    st.title("📊 Predict Employee Income")

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

    if st.button("🎯 Predict Income"):
        prediction = model.predict(input_df)[0]
        income_label = label_encoders['income'].inverse_transform([prediction])[0]
        st.success(f"🧾 Predicted Income: {income_label}")

    st.markdown("---")
    if st.button("⬅️ Back to Home"):
        st.switch_page("?page=home")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Optional banner image
# st.image("salary_banner.jpg", use_column_width=True)

# App title with emoji icon
st.title("💼 Employee Income Prediction App")
st.markdown("Predict whether an employee earns **more than $50K** based on their profile.")

# Load data
data = pd.read_csv("adult 3.csv")

# Remove 'educational-num' if it exists
if 'educational-num' in data.columns:
    data.drop('educational-num', axis=1, inplace=True)

# Clean " ?" and drop missing
data.replace(' ?', pd.NA, inplace=True)
data.dropna(inplace=True)

# Store original text values for dropdowns
original_values = {}
for col in data.select_dtypes(include='object').columns:
    original_values[col] = sorted([val for val in data[col].unique() if val.strip() != '?'])

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Prepare training data
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Input form for user data
def user_input():
    age = st.number_input("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", original_values['workclass'])
    education = st.selectbox("Education", original_values['education'])
    marital_status = st.selectbox("Marital Status", original_values['marital-status'])
    occupation = st.selectbox("Occupation", original_values['occupation'])
    relationship = st.selectbox("Relationship", original_values['relationship'])
    race = st.selectbox("Race", original_values['race'])
    gender = st.selectbox("Gender", original_values['gender'])
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", original_values['native-country'])

    input_data = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'fnlwgt': 200000,  # Fixed example
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

# Predict button
input_df = user_input()

if st.button("🔮 Predict Income"):
    prediction = model.predict(input_df)[0]
    result = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"**Predicted Income**: `{result}`")

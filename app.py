import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("adult 3.csv")

# Drop unnecessary columns
if 'educational-num' in data.columns:
    data.drop('educational-num', axis=1, inplace=True)

# Clean missing values
data.replace(' ?', pd.NA, inplace=True)
data.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Prepare data for training
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Employee Income Prediction")
st.write("Fill in the employee details:")

def user_input():
    age = st.number_input("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
    education = st.selectbox("Education", label_encoders['education'].classes_)
    marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
    occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
    relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
    race = st.selectbox("Race", label_encoders['race'].classes_)
    gender = st.selectbox("Gender", label_encoders['gender'].classes_)
    hours_per_week = st.slider("Hours per week", 1, 99, 40)
    native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)

    # Convert input to encoded dataframe
    data_input = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'fnlwgt': 200000,  # fixed value or optional user input
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
    return pd.DataFrame([data_input])

# Make prediction
input_df = user_input()

if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]
    result = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"Predicted Income: {result}")

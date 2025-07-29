import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("adult 3.csv")

# Drop unnecessary columns if present
if 'education-num' in data.columns:
    data.drop('education-num', axis=1, inplace=True)

# Clean data
data = data.replace(' ?', pd.NA).dropna()

# Encode categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split into train/test
X = data.drop('salary', axis=1)
y = data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Employee Salary Prediction")
st.write("Enter employee details to predict if salary is >50K or <=50K")

# User Input Form
def user_input():
    age = st.number_input('Age', 18, 100, 30)
    workclass = st.selectbox('Workclass', label_encoders['workclass'].classes_)
    education = st.selectbox('Education', label_encoders['education'].classes_)
    marital_status = st.selectbox('Marital Status', label_encoders['marital-status'].classes_)
    occupation = st.selectbox('Occupation', label_encoders['occupation'].classes_)
    relationship = st.selectbox('Relationship', label_encoders['relationship'].classes_)
    race = st.selectbox('Race', label_encoders['race'].classes_)
    sex = st.selectbox('Sex', label_encoders['sex'].classes_)
    hours_per_week = st.number_input('Hours per Week', 1, 100, 40)
    native_country = st.selectbox('Native Country', label_encoders['native-country'].classes_)

    # Convert to model input
    user_data = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'education': label_encoders['education'].transform([education])[0],
        'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
        'occupation': label_encoders['occupation'].transform([occupation])[0],
        'relationship': label_encoders['relationship'].transform([relationship])[0],
        'race': label_encoders['race'].transform([race])[0],
        'sex': label_encoders['sex'].transform([sex])[0],
        'hours-per-week': hours_per_week,
        'native-country': label_encoders['native-country'].transform([native_country])[0],
    }
    return pd.DataFrame([user_data])

input_df = user_input()

if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    result = label_encoders['salary'].inverse_transform([prediction])[0]
    st.success(f"Predicted Salary: {result}")

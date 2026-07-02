import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")

st.title("Cardiovascular Risk Prediction")
st.write("Enter patient details to predict 10-year CHD risk")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox('Sex', ['M', 'F'])
    age = st.number_input('Age', min_value=1, max_value=120, value=50)
    education = st.selectbox('Education Level', [1.0, 2.0, 3.0, 4.0])
    cigsPerDay = st.number_input('Cigarettes Per Day', min_value=0.0, value=0.0)
    BPMeds = st.selectbox('On BP Medication', [0.0, 1.0])
    prevalentStroke = st.selectbox('Prevalent Stroke', [0.0, 1.0])
    diabetes = st.selectbox('Diabetes', [0.0, 1.0])

with col2:
    totChol = st.number_input('Total Cholesterol', min_value=0.0, value=200.0)
    sysBP = st.number_input('Systolic BP', min_value=0.0, value=120.0)
    BMI = st.number_input('BMI', min_value=0.0, value=25.0)
    heartRate = st.number_input('Heart Rate', min_value=0.0, value=75.0)
    glucose = st.number_input('Glucose', min_value=0.0, value=80.0)

if st.button('Predict Risk'):
    data = {
    'age': age,
    'education': education,
    'sex': sex,
    'cigsPerDay': cigsPerDay,
    'BPMeds': BPMeds,
    'prevalentStroke': prevalentStroke,
    'diabetes': diabetes,
    'totChol': totChol,
    'sysBP': sysBP,
    'BMI': BMI,
    'heartRate': heartRate,
    'glucose': glucose
    }

    try:
        response = requests.post('API_URL', json=data)
        st.write(response)
        # st.write(response.text)
        result = response.json()
        prob = result['risk_probability']
        risk = result['risk']
        if risk == 'High Risk':
            st.error(f"{risk} — {prob*100:.1f}% probability of CHD in 10 years")
        else:
            st.success(f"{risk} — {prob*100:.1f}% probability of CHD in 10 years")

    except Exception as e:
        st.write(response.status_code)
        st.write(response.text)
        st.error(f"API error: {e}")
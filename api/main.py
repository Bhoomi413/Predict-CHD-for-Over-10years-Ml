from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

EXPECTED_COLUMNS = [
    'age', 'education', 'sex', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'diabetes', 'totChol', 'sysBP', 
    'BMI', 'heartRate', 'glucose'
]

app = FastAPI()
pipeline= joblib.load('models/logisticRegression_pipe.pkl')
class PatientData(BaseModel):
    age: float
    education: float
    sex: str
    cigsPerDay: float
    BPMeds: float
    prevalentStroke: float
    diabetes: float
    totChol: float
    sysBP: float
    BMI: float
    heartRate: float
    glucose: float

@app.post('/predict')
def predict(data: PatientData):
    try:
        df = pd.DataFrame([data.model_dump()])[EXPECTED_COLUMNS]
        prob = pipeline.predict_proba(df)[0][1]
        risk = 'High Risk' if prob >= 0.5 else 'Low Risk'
        return {'risk_probability': round(float(prob), 3), 'risk': risk}
    except Exception as e:
        return {"error": str(e)}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
pipeline= joblib.load('models/logisticRegression_pipe.pkl')
print(pipeline.feature_names_in_) 
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
    df = pd.DataFrame([data.model_dump()])
    prob = pipeline.predict_proba(df)[0][1]
    risk = 'High Risk' if prob >= 0.5 else 'Low Risk'
    return {'risk_probability': round(float(prob), 3), 'risk': risk}
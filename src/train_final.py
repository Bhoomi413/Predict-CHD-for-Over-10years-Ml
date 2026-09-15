import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import IQROutlierClipper
import joblib


df = pd.read_csv('data/cardiovascular_risk_data.csv', na_values=["NA", "N/A", "null", "None", "?", "-"])

df = df.drop(columns=['id', 'is_smoking', 'diaBP', 'prevalentHyp'])

X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']


numeric_features = ['age', 'cigsPerDay', 'totChol','sysBP', 'BMI', 'heartRate', 'glucose']

# Separate categorical variables by theire specific encoding and imputation needs
education_feature=['education']
sex_feature=['sex']
binary_features= ['BPMeds', 'prevalentStroke', 'diabetes']

#numeric pipeline: imputation -> IQR clipping -> scaling
numeric_transformer =ImbPipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('clipper',IQROutlierClipper(factor=1.5)),
    ('scaler',StandardScaler())
])

#education pipeline: impute missing -> OneHotEncode
education_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(categories=[[1.0, 2.0, 3.0, 4.0]], drop=[4.0], handle_unknown='ignore'))
])



#sex pipeline: impute missing -> OneHotEncoder
sex_transformer =ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# binary features pipeline: impute missing -> passthrough
binary_transformer=ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

#ColumnTransformer
preprocessor =ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('edu', education_transformer, education_feature),
    ('sex', sex_transformer, sex_feature),
    ('binary', binary_transformer, binary_features)
])


# complete pipeline
pipeline=ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
    ('classifier', LogisticRegression(class_weight='balanced', C=1, max_iter=1000, random_state=42))
])

# Fit the pipeline on entire dataset bcse we already benchmarked models
pipeline.fit(X, y)
joblib.dump(pipeline, "models/logisticRegression_pipe.pkl")

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from src.preprocess import load_data, impute_glucose, clip_outliers, encode_education, encode_sex



df = load_data('data/cardiovascular_risk_data.csv')


X = df.drop(columns='TenYearCHD')     # independent features
y = df['TenYearCHD']                  # dependent features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)


pipeline = Pipeline(steps=[
    ('impute_glucose', FunctionTransformer(impute_glucose)),
    ('clip_outliers', FunctionTransformer(clip_outliers)),
    ('encode_education', FunctionTransformer(encode_education)),
    ('encode_sex', FunctionTransformer(encode_sex)),
    ('scale', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', C=1, max_iter=1000))
])
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "models/logisticRegression_pipe.pkl")


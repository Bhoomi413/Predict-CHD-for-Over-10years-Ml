from sklearn.impute import KNNImputer
import pandas as pd

def get_feature_types(df):
    numeric    = [c for c in df.columns if df[c].nunique() > 10]
    categorical = [c for c in df.columns if df[c].nunique() <= 10]
    return numeric, categorical

def load_data(path):
    df = pd.read_csv(path, na_values=["NA", "N/A", "null", "None", "?", "-"])
    df.drop(['id', 'is_smoking', 'diaBP', 'prevalentHyp'], axis=1, inplace=True, errors='ignore')
    df.dropna(subset=['education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate'], inplace=True)
    return df

from sklearn.impute import KNNImputer
def impute_glucose(X):
    X = X.copy()
    knn = KNNImputer(weights='distance')
    X['glucose'] = knn.fit_transform(X[['glucose']])
    return X

def clip_outliers(X):
    X = X.copy()
    numeric_features, _ = get_feature_types(X)
    for col in numeric_features:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower_bound, upper_bound)
    return X

def encode_education(X):
    X = X.copy()
    dummies = pd.get_dummies(X['education'])
    dummies.columns = ['education_1', 'education_2', 'education_3', 'education_4']
    dummies = dummies.astype(int) 
    X = pd.concat([X, dummies], axis=1)
    X = X.drop(['education', 'education_4'], axis=1, errors='ignore')
    return X

def encode_sex(X):
    X = X.copy()
    X['sex'] = X['sex'].map({'M': 1, 'F': 0}) 
    return X
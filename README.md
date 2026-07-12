---
title: Cardio Risk API
emoji: 🫀
sdk: docker
app_port: 7860
---


### predicts that patient is at risk for coronary heart disease in coming ten years | Machine Learning Project

### Live Demo
- App: https://predict-cardio-risk-ml.streamlit.app/

## Tech Stack
- **ML:** scikit-learn, imbalanced-learn (SMOTE), missingno
- **API:** FastAPI
- **Deployment:** Hugging Face Spaces (API), Streamlit Community Cloud (UI)
- **Containerisation:** Docker
- **Frontend:** Streamlit

## ML Pipeline
- Dropped correlated features: `diaBP`, `prevalentHyp`, `is_smoking` (EDA insights)
- KNN imputation for missing glucose values, Complete Case Analysis (CCA) for columns containing missing values less than 5% were removed to analyze the dataset without missing data.
- IQR-based outlier clipping
- One-hot encoding for education, binary encoding for sex
- SMOTE for class imbalance (sampling_strategy=0.3)
- Final model: Logistic Regression (class_weight='balanced', C=1)

Note: All preprocessing functions are centralised in `src/preprocess.py` and imported across notebooks and training script — avoiding code duplication.

### Run API locally
```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
```

### Run Streamlit locally
```bash
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

### Results

| Model | Test ROC-AUC | Test Recall |
|-------|-------------:|------------:|
| Logistic Regression | 0.730 | 0.705 |
| SVM | 0.597 | 0.091 |
| Random Forest | 0.700 | 0.182 |
| XGBoost | 0.678 | 0.136 |
| KNN | 0.528 | 0.193 |
| DecisionTree | 0.544 | 0.250 |

## Project Structure

```text
cardiovascular-risk/
├── Dockerfile                     # FastAPI container (HF Spaces)
├── requirements-api.txt           # API dependencies
├── requirements.txt               # Streamlit dependencies
├── api/
│   └── main.py                    # FastAPI /predict endpoint
├── app/
│   └── streamlit_app.py           # Streamlit frontend
├── src/
│   ├── preprocess.py              # Data preprocessing functions
│   └── train.py                   # Model training
├── notebooks/
│   ├── EDA_notebook.ipynb         # Exploratory Data Analysis (EDA)
│   └── compareModels.ipynb        # Model comparison
├── data/
│   └── cardiovascular_risk_data.csv
├── .gitignore                     # Git ignore rules
└── .dockerignore                  # Docker ignore rules
```


### Dataset

The data comes from an ongoing cardiovascular study of Framingham, Massachusetts, residents. The purpose of the classification is to determine whether the patient is at risk for coronary heart disease (CHD) in the ten years to come.

### Explorartory Data Analysis

- A statistical summary of the dataset was generated.
- Missing values were visualized using the missingno library.
- Understanding Distribution of numeric variables using Histogram and Boxplot.
- Countplot helped understand Distribution of categorical variables-
Univariate Analysis of Categorical Features
Bivariate Analysis of Categorical Features
- Plotting correlation heatmap and making a function to find Multicollinearity between features.
- code file - notebooks/EDA_notebook.ipynb

### Comparing Various Models

- Devloped a reusable function to evaluate models using roc_auc_score, recall, f1_score, precision, avg precison, accuracy and plotting confusion matrix and ROC curve for each model.
- performing Hyperparameter Tunning using RandomizedSearchCV, GridSearchCV.
- plotting graph to compaire model performance of all the models
- code file - notebooks/compareModels.ipynb

# Experiments
SMOTE-Tomek was evaluated but did not improve performance due to dataset size and distribution sensitivity; SMOTE alone provided better generalization

# Chose Logistic Regression
Our model identified 62 out of 88 high-risk patients correctly. The 171 cases flagged as high-risk but testing negative represent individuals who would receive further clinical screening — a standard and acceptable outcome in preventive healthcare, where early intervention is prioritized over perfect precision

Logistic Regression demonstrated more consistent performance between the training and test sets, indicating better generalization to unseen data.

Random Forest, and XGBoost achieved very high performance on the training data but showed a significant drop in performance on the test set, indicating overfitting.

Based on this comparison, Logistic Regression was selected as the final model.

## Repository Status & Access

This project is actively used as part of my professional portfolio and B.Tech major project evaluation. 

**Recruiters & Hiring Managers:** You are welcome to clone, explore, and test the RAG pipeline. If you have questions about the system architecture or AI integration, feel free to reach out!
**Students & Peer Reviewers:** This code is indexed for academic evaluation. Please use it for inspiration or reference only. To maintain academic integrity, do not copy or replicate these files for university submissions.

## Disclaimer

This project is intended solely for educational and demonstration purposes. It is not a medical device and does not provide medical advice, diagnosis, or treatment. Predictions generated by this model should not be used for clinical decision-making or as a substitute for consultation with a qualified healthcare professional.
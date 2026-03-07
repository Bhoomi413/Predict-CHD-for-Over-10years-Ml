#  predicts that patient is at risk for coronary heart disease in coming ten years | Machine Learning Project

### The data comes from an ongoing cardiovascular study of Framingham, Massachusetts, residents. The purpose of the classification is to determine whether the patient is at risk for coronary heart disease (CHD) in the ten years to come.

### Initial Data Exploration

A statistical summary of the dataset was generated

histogram helped understand Distribution of numeric variables.

countplot helped understand Distribution of categorical variables.

Missing values were visualized using the missingno library.

### Handling Missing Values

K-Nearest Neighbors (KNN) Imputation : KNN Imputation was used to estimate missing values for glucose column.

Complete Case Analysis (CCA) : Rows containing missing values less than 5% were removed to analyze the dataset without missing data.


### Tools and Libraries

Pandas, Missingno, Scikit-learn, matplotlib, seaborn, numpy
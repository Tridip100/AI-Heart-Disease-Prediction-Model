Overview
This project is a Machine Learning–based Heart Disease Prediction system that evaluates patient clinical and physiological parameters to estimate the risk of heart disease. Multiple supervised learning algorithms were trained and compared to select the most accurate model.

The final deployed application is built using Streamlit with an interactive medical dashboard interface.

Machine Learning Models Implemented
The following classification algorithms were trained and evaluated:

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree

Support Vector Machine (SVM)

Naive Bayes

All models were trained using the same preprocessing pipeline to ensure fair comparison.

Best Performing Model
After evaluating performance metrics:

K-Nearest Neighbors (KNN) achieved the highest accuracy of 86.96%

Therefore, KNN was selected as the final deployed model.

ML Pipeline
Data Cleaning & Preprocessing

Categorical Encoding

Feature Scaling using StandardScaler

Train-Test Split

Model Training

Performance Evaluation

Model Serialization using Joblib

Saved artifacts:

KNN_Heart.pkl

scaler.pkl

columns.pkl

Evaluation Metrics
Models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

KNN demonstrated the strongest generalization performance on unseen test data.
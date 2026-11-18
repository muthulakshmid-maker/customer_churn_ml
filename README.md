# Customer Churn Prediction

## Assumptions
- Each row represents one customer.
- Churn = Yes means the customer left.
- Churn = No means the customer stayed.
- Historical customer information can help predict churn.
- The test dataset remains untouched until final evaluation.

## Evaluation Metric
Primary Metric: F1-score

Reason:
- Churn data may be imbalanced.
- Accuracy alone can be misleading.
- F1-score balances precision and recall.

Secondary Metric: ROC-AUC.










# Customer Churn Prediction Using Machine Learning

## 1. Project Overview
This project predicts whether a telecom customer
is likely to leave the company using machine learning.

## 2. Dataset
Telco Customer Churn Dataset.

## 3. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Git and GitHub

## 4. Project Workflow
1. Data Loading
2. Data Inspection
3. Exploratory Data Analysis
4. Train-Test Split (80:20)
5. Data Preprocessing
6. Feature Engineering
7. Model Training
8. Model Selection
9. Final Model Evaluation
10. Customer Churn Prediction

## 5. Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## 6. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Primary metric: F1-score.

## 7. Model Artifacts
- best_churn_model.pkl
- model_comparison.csv
- final_metrics.csv
- churn_predictions.csv

## 8. How to Run

Install dependencies:

pip install -r requirements.txt

Train models:

python src/model_training.py

Evaluate model:

python src/model_evaluation.py

Predict new customers:

python src/predict_new_customer.py

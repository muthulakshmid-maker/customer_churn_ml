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
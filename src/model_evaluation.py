import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)



model = joblib.load("models/best_churn_model.pkl")

print("Best model loaded successfully!")



test_df = pd.read_csv("data/test_data.csv")

print("\nTest Dataset Shape:", test_df.shape)

test_df["TotalCharges"] = (
    test_df["TotalCharges"]
    .astype(str)
    .str.strip()
)

test_df["TotalCharges"] = pd.to_numeric(
    test_df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges Data Type:")
print(test_df["TotalCharges"].dtype)

print("\nMissing TotalCharges:")
print(test_df["TotalCharges"].isnull().sum())



X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"].map({
    "No": 0,
    "Yes": 1
})



y_pred = model.predict(X_test)


y_prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n========== FINAL MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Customer Churn Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()



RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title("Customer Churn ROC Curve")

plt.savefig(
    "outputs/roc_curve.png",
    bbox_inches="tight"
)

plt.close()



results = test_df.copy()

results["Actual_Churn"] = y_test.map({
    0: "No",
    1: "Yes"
})

results["Predicted_Churn"] = pd.Series(
    y_pred,
    index=results.index
).map({
    0: "No",
    1: "Yes"
})

results["Churn_Probability"] = y_prob

results.to_csv(
    "outputs/churn_predictions.csv",
    index=False
)


metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    "outputs/final_metrics.csv",
    index=False
)

print("\nEvaluation completed successfully!")

print("Results saved in outputs folder.")

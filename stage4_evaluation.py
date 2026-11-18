import os
import joblib
import numpy as np
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



os.makedirs("outputs", exist_ok=True)

print("\nCUSTOMER CHURN PREDICTION")
print("STAGE 4: FINAL MODEL EVALUATION")
print("=" * 60)



model = joblib.load(
    "models/best_churn_model.pkl"
)

print("\nTrained model loaded successfully!")



X_test = joblib.load(
    "outputs/X_test_raw.pkl"
)

y_test = joblib.load(
    "outputs/y_test.pkl"
)

print("\nTest data loaded successfully!")

print("Test features shape:", X_test.shape)
print("Test target shape:", y_test.shape)



X_test = X_test.drop(
    columns=["customerID"]
)


X_test["TotalCharges"] = pd.to_numeric(
    X_test["TotalCharges"],
    errors="coerce"
)


X_test["AverageMonthlySpend"] = (
    X_test["TotalCharges"] /
    X_test["tenure"].replace(0, np.nan)
)


X_test["TenureGroup"] = pd.cut(
    X_test["tenure"],
    bins=[-1, 12, 24, 48, np.inf],
    labels=[
        "New",
        "Medium",
        "LongTerm",
        "Loyal"
    ]
)


service_columns = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

X_test["ServiceCount"] = (
    X_test[service_columns] == "Yes"
).sum(axis=1)

print("\nFeature engineering completed!")



y_pred = model.predict(X_test)


y_probability = model.predict_proba(X_test)[:, 1]

print("\nPredictions generated successfully!")



accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nFINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")



print("\nCLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Churn",
        "Churn"
    ],
    zero_division=0
)

print(report)

with open(
    "outputs/classification_report.txt",
    "w"
) as file:

    file.write(report)



cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Churn",
        "Churn"
    ],
    yticklabels=[
        "No Churn",
        "Churn"
    ]
)

plt.title("Customer Churn Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()



plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Customer Churn ROC Curve")

plt.savefig(
    "outputs/roc_curve.png"
)

plt.show()



comparison = pd.DataFrame({

    "Actual_Churn": y_test.values,

    "Predicted_Churn": y_pred,

    "Churn_Probability": y_probability

}, index=X_test.index)



false_positives = comparison[
    (comparison["Actual_Churn"] == 0) &
    (comparison["Predicted_Churn"] == 1)
]



false_negatives = comparison[
    (comparison["Actual_Churn"] == 1) &
    (comparison["Predicted_Churn"] == 0)
]

print("\nFALSE POSITIVES:", len(false_positives))

print("FALSE NEGATIVES:", len(false_negatives))

false_positives.to_csv(
    "outputs/false_positives.csv",
    index=True
)

false_negatives.to_csv(
    "outputs/false_negatives.csv",
    index=True
)



predictions = pd.DataFrame({

    "Actual_Churn": y_test.values,

    "Predicted_Churn": y_pred,

    "Churn_Probability": y_probability

}, index=X_test.index)

predictions["Actual_Label"] = (
    predictions["Actual_Churn"].map({
        0: "No",
        1: "Yes"
    })
)

predictions["Predicted_Label"] = (
    predictions["Predicted_Churn"].map({
        0: "No",
        1: "Yes"
    })
)

predictions.to_csv(
    "outputs/churn_predictions.csv",
    index=True
)

print("\nPredictions saved successfully!")



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



print("\n" + "=" * 60)

print("FINAL PROJECT EVALUATION COMPLETED")

print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nGENERATED FILES:")

print("outputs/confusion_matrix.png")
print("outputs/roc_curve.png")
print("outputs/classification_report.txt")
print("outputs/churn_predictions.csv")
print("outputs/final_metrics.csv")
print("outputs/false_positives.csv")
print("outputs/false_negatives.csv")

print("\nCUSTOMER CHURN PROJECT COMPLETED!")

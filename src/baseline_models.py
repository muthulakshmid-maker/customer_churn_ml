import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate



df = pd.read_csv("data/train_data.csv")

print("Original Dataset Shape:", df.shape)

df = df.drop(columns=["customerID"], errors="ignore")

df["TotalCharges"] = (
    df["TotalCharges"].astype(str).str.strip()
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)


df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["New", "Short", "Medium", "Long"]
)


df["spend_per_month"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, np.nan)
)


service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

def count_services(row):
    count = 0

    for col in service_columns:
        if row[col] == "Yes":
            count += 1

    if row["InternetService"] != "No":
        count += 1

    return count

df["total_services"] = df.apply(
    count_services, axis=1
)


df["is_long_term"] = df["Contract"].isin(
    ["One year", "Two year"]
).astype(int)


df["has_family"] = (
    (df["Partner"] == "Yes") |
    (df["Dependents"] == "Yes")
).astype(int)


df["charge_per_service"] = (
    df["MonthlyCharges"] /
    df["total_services"].replace(0, np.nan)
)

print("\nNew Features Created:")
print([
    "tenure_group",
    "spend_per_month",
    "total_services",
    "is_long_term",
    "has_family",
    "charge_per_service"
])

print("\nUpdated Dataset Shape:", df.shape)


X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected Churn target values.")



numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("\nCategorical Features:", categorical_features)



numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)



models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
}



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1"
}

results = []



for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    mean_roc_auc = scores["test_roc_auc"].mean()
    mean_f1 = scores["test_f1"].mean()

    print("CV ROC-AUC:", round(mean_roc_auc, 4))
    print("CV F1:", round(mean_f1, 4))

    results.append({
        "Model": name,
        "CV ROC-AUC": round(mean_roc_auc, 4),
        "CV F1": round(mean_f1, 4)
    })



results_df = pd.DataFrame(results)

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/baseline_results.csv",
    index=False
)

print("\n========== BASELINE RESULTS ==========")
print(results_df.to_string(index=False))

print("\nResults saved to outputs/baseline_results.csv")

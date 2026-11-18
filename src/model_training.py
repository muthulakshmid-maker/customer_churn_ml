import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate


df = pd.read_csv("data/train_data.csv")

print("Training Dataset Shape:", df.shape)


df = df.drop(columns=["customerID"], errors="ignore")

df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


print("\nMissing Values After Cleaning:")

print(df.isnull().sum())


print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

print("\nTotalCharges Missing Values:")
print(df["TotalCharges"].isnull().sum())


X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn target.")



numerical_features = X.select_dtypes(
    include=["number"]
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

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}



cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall"
}

results = []

best_score = -1
best_name = None
best_pipeline = None



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
        n_jobs=-1,
        return_train_score=False
    )

    mean_f1 = scores["test_f1"].mean()
    mean_auc = scores["test_roc_auc"].mean()
    mean_precision = scores["test_precision"].mean()
    mean_recall = scores["test_recall"].mean()

    print("Mean F1-score:", round(mean_f1, 4))
    print("Mean ROC-AUC:", round(mean_auc, 4))
    print("Mean Precision:", round(mean_precision, 4))
    print("Mean Recall:", round(mean_recall, 4))

    results.append({
        "Model": name,
        "F1 Score": mean_f1,
        "ROC-AUC": mean_auc,
        "Precision": mean_precision,
        "Recall": mean_recall
    })

    if mean_f1 > best_score:
        best_score = mean_f1
        best_name = name
        best_pipeline = pipeline



results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)



print("\nBest Model:", best_name)
print("Best Cross-validation F1:", round(best_score, 4))

best_pipeline.fit(X, y)


os.makedirs("models", exist_ok=True)

joblib.dump(
    best_pipeline,
    "models/best_churn_model.pkl"
)

print("\nBest model saved successfully!")
print("Saved at: models/best_churn_model.pkl")

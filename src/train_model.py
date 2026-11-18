import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")



os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("\nCUSTOMER CHURN PREDICTION PROJECT")
print("=" * 55)



df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("\nDATASET LOADED SUCCESSFULLY")



print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nData Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nClass Balance:")
print(df["Churn"].value_counts())

print("\nClass Balance Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)



X = df.drop(columns=["Churn"])
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAIN TEST SPLIT")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)




eda_df = X_train.copy()
eda_df["Churn"] = y_train

sns.set_theme(style="whitegrid")


plt.figure(figsize=(7, 5))

sns.countplot(
    data=eda_df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution")
plt.savefig("outputs/chart1_churn_distribution.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=eda_df,
    x="Contract",
    hue="Churn",
    palette="Set1"
)

plt.title("Churn by Contract Type")
plt.savefig("outputs/chart2_contract.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=eda_df,
    x="Churn",
    y="tenure",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Tenure vs Churn")
plt.savefig("outputs/chart3_tenure.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=eda_df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set3",
    legend=False
)

plt.title("Monthly Charges vs Churn")
plt.savefig("outputs/chart4_monthly_charges.png")
plt.show()


plt.figure(figsize=(8, 5))

sns.countplot(
    data=eda_df,
    x="InternetService",
    hue="Churn",
    palette="Set2"
)

plt.title("Internet Service vs Churn")
plt.savefig("outputs/chart5_internet_service.png")
plt.show()

print("\nEDA COMPLETED")



X_train = X_train.drop(columns=["customerID"])


X_train["TotalCharges"] = pd.to_numeric(
    X_train["TotalCharges"],
    errors="coerce"
)



X_train["AverageMonthlySpend"] = (
    X_train["TotalCharges"] /
    X_train["tenure"].replace(0, np.nan)
)


X_train["TenureGroup"] = pd.cut(
    X_train["tenure"],
    bins=[-1, 12, 24, 48, np.inf],
    labels=["New", "Medium", "LongTerm", "Loyal"]
)

# Number of subscribed services

service_columns = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

X_train["ServiceCount"] = (
    X_train[service_columns] == "Yes"
).sum(axis=1)

print("\nFEATURE ENGINEERING COMPLETED")



numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)



numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

print("\nPREPROCESSING PIPELINE CREATED")



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
    )

}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc"
}

results = []

print("\nSTARTING MODEL TRAINING")



for name, model in models.items():

    print(f"\nTraining: {name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({

        "Model": name,

        "F1": scores["test_f1"].mean(),

        "Precision": scores["test_precision"].mean(),

        "Recall": scores["test_recall"].mean(),

        "ROC_AUC": scores["test_roc_auc"].mean()

    })



results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\nMODEL COMPARISON")
print("=" * 55)

print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)



best_model_name = results_df.iloc[0]["Model"]

print("\nBEST MODEL:", best_model_name)



best_model = models[best_model_name]

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_model)
    ]
)

best_pipeline.fit(
    X_train,
    y_train
)

print("\nBEST MODEL TRAINED SUCCESSFULLY")


if best_model_name == "Random Forest":

    print("\nSTARTING RANDOM FOREST TUNING")

    tuning_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            ))
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }

    search = RandomizedSearchCV(
        tuning_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_

    print("\nBEST PARAMETERS:")
    print(search.best_params_)

    print("\nBEST CROSS VALIDATION F1:")
    print(search.best_score_)

else:

    print("\nRandom Forest was not the winning model.")
    print("Keeping the highest F1 model without tuning.")



joblib.dump(
    best_pipeline,
    "models/best_churn_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY")



joblib.dump(
    X_train,
    "outputs/X_train_raw.pkl"
)

joblib.dump(
    y_train,
    "outputs/y_train.pkl"
)

joblib.dump(
    X_test,
    "outputs/X_test_raw.pkl"
)

joblib.dump(
    y_test,
    "outputs/y_test.pkl"
)

print("\nTRAINING AND RESERVED TEST DATA SAVED")



print("\n" + "=" * 55)

print("CUSTOMER CHURN PROJECT - STAGE 3 COMPLETED")

print("=" * 55)

print("Selected Model:", best_model_name)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\nSaved Files:")

print("models/best_churn_model.pkl")
print("models/preprocessor.pkl")
print("outputs/model_comparison.csv")

print("\nThe held-out test data has NOT been used for evaluation.")

print("\nREADY FOR STAGE 4: FINAL MODEL EVALUATION")

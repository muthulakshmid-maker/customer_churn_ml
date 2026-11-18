import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



df = pd.read_csv("data/train_data.csv")

print("Original Shape:", df.shape)



df = df.drop(columns=["customerID"], errors="ignore")


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)



X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nTarget Distribution:")
print(y.value_counts())



numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)



numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])



categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    ))
])



preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)



X_processed = preprocessor.fit_transform(X)

print("\nPreprocessing Completed!")

print("Original Feature Shape:", X.shape)
print("Processed Feature Shape:", X_processed.shape)


joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nPreprocessor saved successfully!")

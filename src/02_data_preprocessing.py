import os
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)



print("\n========== LOADING TRAINING DATA ==========\n")

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "X_train.csv")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
).squeeze("columns")

print("Training Features Shape:", X_train.shape)
print("Training Target Shape:", y_train.shape)

print("\nFirst Five Records:")
print(X_train.head())

print("\nTarget Distribution:")
print(y_train.value_counts())




print("\n========== DATA INSPECTION ==========\n")

print("Data Types:")
print(X_train.dtypes)

print("\nMissing Values:")
print(X_train.isnull().sum())

print("\nTotal Missing Values:")
print(X_train.isnull().sum().sum())




print("\n========== FEATURE IDENTIFICATION ==========\n")

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nTotal Numerical Features:",
      len(numerical_features))

print("Total Categorical Features:",
      len(categorical_features))



print("\n========== NUMERICAL PREPROCESSING ==========\n")

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

print("Numerical Pipeline Created Successfully!")



print("\n========== CATEGORICAL PREPROCESSING ==========\n")

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

print("Categorical Pipeline Created Successfully!")




print("\n========== COLUMN TRANSFORMER ==========\n")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

print("ColumnTransformer Created Successfully!")



print("\n========== FITTING PREPROCESSOR ==========\n")

X_train_processed = preprocessor.fit_transform(
    X_train
)

print("Training Data Preprocessed Successfully!")

print("Processed Shape:", X_train_processed.shape)




print("\n========== FEATURE NAMES ==========\n")

feature_names = preprocessor.get_feature_names_out()

print("Total Features After Encoding:", len(feature_names))

for feature in feature_names:
    print(feature)



print("\n========== CONVERTING TO DATAFRAME ==========\n")

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

print(X_train_processed_df.head())

print("\nProcessed Data Shape:",
      X_train_processed_df.shape)




print("\n========== VALIDATION ==========\n")

print(
    "Missing Values:",
    X_train_processed_df.isnull().sum().sum()
)

print(
    "Infinite Values:",
    np.isinf(X_train_processed_df.to_numpy()).sum()
)

print(
    "All Numeric:",
    X_train_processed_df.select_dtypes(
        exclude=np.number
    ).empty
)



print("\n========== SAVING PROCESSED DATA ==========\n")

X_train_processed_df.to_csv(
    os.path.join(
        DATA_DIR,
        "X_train_processed.csv"
    ),
    index=False
)

y_train.to_csv(
    os.path.join(
        DATA_DIR,
        "y_train_processed.csv"
    ),
    index=False
)

print("Processed Training Data Saved!")



print("\n========== SAVING PREPROCESSOR ==========\n")

preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)

print("Preprocessor Saved Successfully!")
print("Saved Location:", preprocessor_path)



feature_path = os.path.join(
    REPORT_DIR,
    "processed_feature_names.txt"
)

with open(feature_path, "w") as file:

    for feature in feature_names:
        file.write(str(feature) + "\n")

print("Feature Names Saved Successfully!")



print("\n========== PREPROCESSING SUMMARY ==========\n")

print("Original Training Shape:", X_train.shape)

print("Processed Training Shape:",
      X_train_processed_df.shape)

print("Numerical Features:", len(numerical_features))

print("Categorical Features:", len(categorical_features))

print("Missing Value Handling: Median / Most Frequent")

print("Encoding: One-Hot Encoding")

print("Scaling: StandardScaler")

print("Pipeline: ColumnTransformer + Pipeline")

print("Test Data Used: NO")

print("\nSTAGE 2 COMPLETED SUCCESSFULLY!")

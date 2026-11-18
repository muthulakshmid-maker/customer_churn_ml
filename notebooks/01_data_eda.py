import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_STATE = 42

RANDOM_STATE = 42

sns.set_theme(style="whitegrid")

print("\n========== LOADING DATASET ==========\n")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("\nFirst 5 Records:")
print(df.head())


print("\n========== DATASET INFORMATION ==========\n")

print("Dataset Shape:", df.shape)

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])



print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Summary:")
print(df.describe(include="object"))


print("\n========== MISSING VALUE ANALYSIS ==========\n")

missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_report = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_report)


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing values after converting TotalCharges:")

print(df.isnull().sum())


print("\n========== DUPLICATE ANALYSIS ==========\n")

duplicate_count = df.duplicated().sum()

print("Total Duplicate Rows:", duplicate_count)

# Check duplicate customer IDs
print(
    "Duplicate Customer IDs:",
    df["customerID"].duplicated().sum()
)


print("\n========== TARGET CLASS BALANCE ==========\n")

print("Churn Counts:")

print(df["Churn"].value_counts())

print("\nChurn Percentages:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)




print("\n========== BASIC DATA CLEANING ==========\n")


df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("Target variable converted successfully.")

print(df["Churn"].value_counts())


print("\n========== TRAIN TEST SPLIT ==========\n")


X = df.drop(columns=["Churn"])

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)



print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


train_df = X_train.copy()

train_df["Churn"] = y_train


print("\n========== EXPLORATORY DATA ANALYSIS ==========\n")


plt.figure(figsize=(8, 5))

sns.countplot(
    data=train_df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "01_churn_distribution.png")
)

plt.show()


plt.figure(figsize=(9, 5))

sns.countplot(
    data=train_df,
    x="Contract",
    hue="Churn",
    palette="Set1"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")


plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "02_churn_by_contract.png")
)

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=train_df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "03_monthly_charges_vs_churn.png")
)

plt.show()


plt.figure(figsize=(9, 5))

sns.histplot(
    data=train_df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    element="step",
    stat="density",
    common_norm=False
)


plt.title("Customer Tenure vs Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Density")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "04_tenure_vs_churn.png")
)

plt.show()


plt.figure(figsize=(9, 5))

sns.countplot(
    data=train_df,
    x="InternetService",
    hue="Churn",
    palette="Set2"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service Type")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "05_internet_service_vs_churn.png")
)

plt.show()

print("\n========== EDA OBSERVATIONS ==========\n")

print("""
1. Churn Distribution:
   Check whether the target variable is imbalanced.

2. Contract Type:
   Month-to-month customers may have higher churn.

3. Monthly Charges:
   Compare monthly charges between churned and
   non-churned customers.

4. Tenure:
   Customers with shorter tenure may be more
   likely to leave.

5. Internet Service:
   Investigate differences in churn across
   DSL, Fiber optic and No internet service.

   
Note:
These are hypotheses to investigate, not confirmed
findings. Use the generated charts to record actual
observations.
""")



print("\n========== PROJECT ASSUMPTIONS ==========\n")

assumptions = [
    "The dataset contains historical telecom customer data.",
    "Churn is the target variable.",
    "Yes represents customers who left.",
    "No represents customers who stayed.",
    "CustomerID is an identifier and will be excluded.",
    "TotalCharges blank strings will be treated as missing.",
    "Missing values will be handled during preprocessing.",
    "Training data will be used for EDA and model development.",
    "Test data will remain untouched until final evaluation."
]

for index, assumption in enumerate(assumptions, start=1):
    print(f"{index}. {assumption}")


print("\n========== EVALUATION METRIC ==========\n")

print("""
Primary Optimization Metric: F1-Score

Reason:
- Churn datasets can be imbalanced.
- Recall is important because we want to identify
  customers who are likely to leave.
- Precision is important because unnecessary
  retention campaigns can cost money.
- F1-score balances precision and recall.


Secondary Metric: ROC-AUC

Reason:
- Measures how well the model separates churn
  and non-churn customers across thresholds.

Final test evaluation will happen only after
model development and tuning are complete.
""")



print("\n========== SAVING DATASETS ==========\n")

X_train.to_csv(
    os.path.join(PROCESSED_DIR, "X_train.csv"),
    index=False
)

y_train.to_csv(
    os.path.join(PROCESSED_DIR, "y_train.csv"),
    index=False
)

X_test.to_csv(
    os.path.join(PROCESSED_DIR, "X_test.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(PROCESSED_DIR, "y_test.csv"),
    index=False
)

print("Training and test datasets saved successfully!")


print("\n========== PROJECT STAGE 1 COMPLETED ==========\n")

print("Dataset Shape:", df.shape)

print("Training Records:", len(X_train))

print("Testing Records:", len(X_test))

print("EDA Charts Generated: 5")

print("Primary Metric: F1-Score")

print("Secondary Metric: ROC-AUC")

print("\nGenerated Charts Location:")
print(FIGURE_DIR)

print("\nProcessed Data Location:")
print(PROCESSED_DIR)

print("\nStage 1: Data Loading and EDA completed successfully!")

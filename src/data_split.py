import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


X = df.drop("Churn", axis=1)
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


train_df = X_train.copy()
train_df["Churn"] = y_train

train_df.to_csv("data/train_data.csv", index=False)

test_df = X_test.copy()
test_df["Churn"] = y_test

test_df.to_csv("data/test_data.csv", index=False)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

print("Train-test split completed.")

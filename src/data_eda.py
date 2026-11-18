import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("data/train_data.csv")


os.makedirs("outputs/eda_charts", exist_ok=True)

sns.set_theme(style="whitegrid")


plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)
plt.title("Customer Churn Distribution")
plt.savefig("outputs/eda_charts/01_churn_distribution.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set1"
)
plt.title("Churn vs Contract Type")
plt.savefig("outputs/eda_charts/02_contract_churn.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn",
    palette="Set2"
)
plt.title("Churn vs Internet Service")
plt.savefig("outputs/eda_charts/03_internet_churn.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="tenure",
    hue="Churn",
    palette="Set3",
    legend=False
)
plt.title("Churn vs Tenure")
plt.savefig("outputs/eda_charts/04_tenure_churn.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)
plt.title("Churn vs Monthly Charges")
plt.savefig("outputs/eda_charts/05_monthly_charges.png")
plt.close()

print("All 5 EDA charts generated successfully!")

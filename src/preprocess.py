import pandas as pd
import os

print("Preprocessing data...")

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/titanic.csv")

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop useless columns
df.drop(["Cabin"], axis=1, inplace=True)

# Encode categorical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"])

df.to_csv("data/processed/processed.csv", index=False)

print("✅ Preprocessed data → data/processed/processed.csv")

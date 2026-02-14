import pandas as pd
import os

print("Creating features...")

os.makedirs("features", exist_ok=True)

df = pd.read_csv("data/processed/processed.csv")

# Create family size feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Remove non useful columns
df.drop(["Name", "Ticket", "PassengerId"], axis=1, errors="ignore", inplace=True)

df.to_csv("features/features.csv", index=False)

print("✅ Features saved → features/features.csv")

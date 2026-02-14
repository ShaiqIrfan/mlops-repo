import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Training model...")

os.makedirs("models", exist_ok=True)

df = pd.read_csv("features/features.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved → models/model.pkl")

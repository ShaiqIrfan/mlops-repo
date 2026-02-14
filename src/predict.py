import pandas as pd
import pickle
import os

print("Generating predictions...")

os.makedirs("result", exist_ok=True)

df = pd.read_csv("features/features.csv")

X = df.drop("Survived", axis=1)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X)

pd.DataFrame(predictions, columns=["Prediction"]).to_csv(
    "result/predictions.csv", index=False
)

print("✅ Predictions saved → result/predictions.csv")

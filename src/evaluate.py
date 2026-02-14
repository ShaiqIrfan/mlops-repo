import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Evaluating model...")

df = pd.read_csv("features/features.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(X)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)
f1 = f1_score(y, pred)

with open("result/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

print("✅ Metrics saved → result/metrics.txt")

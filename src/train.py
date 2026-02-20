"""
Model Training Script
- Trains Logistic Regression on preprocessed data
- Saves model in models/
- Prints accuracy
"""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def main():
    print("Loading preprocessed data...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).squeeze()

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Calculate and print accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"\n--- Training Results ---")
    print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to {model_path}")
    print("Training complete!")

if __name__ == "__main__":
    main()

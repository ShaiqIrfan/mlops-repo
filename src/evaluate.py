"""
Model Evaluation Script
- Loads trained model and test data
- Computes and saves evaluation metrics
- Prints final accuracy and classification report
"""

import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def main():
    print("Loading model and test data...")
    with open(os.path.join(MODELS_DIR, 'logistic_regression_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).squeeze()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save results to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))

    print(f"\nResults saved to {results_path}")
    print("Evaluation complete!")

if __name__ == "__main__":
    main()

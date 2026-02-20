"""
Data Preprocessing Script
- Load dataset from CSV
- Handle missing values
- Split into train/test
- Save processed data in data/processed/
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
INPUT_FILE = os.path.join(DATA_DIR, 'iris.csv')

def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")
    print(f"Missing values before:\n{df.isnull().sum()}")

    # Handle missing values - fill with median for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    print(f"\nMissing values after handling:\n{df.isnull().sum()}")

    # Encode target variable
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])

    X = df.drop('species', axis=1)
    y = df['species']

    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create processed directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save processed data
    X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

    # Save label encoder for inference
    pd.Series(le.classes_).to_csv(os.path.join(PROCESSED_DIR, 'classes.csv'), index=False)

    print(f"\nPreprocessed data saved to {PROCESSED_DIR}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()

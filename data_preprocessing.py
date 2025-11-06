# Task 1 - Data Preprocessing (Alfido Tech AI Internship)
# Author: [Your Name]

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Check data info
print("Initial Data Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:, :-1])

# Create scaled dataframe
df_scaled = pd.DataFrame(scaled_features, columns=iris.feature_names)
df_scaled['target'] = df['target']

# Split data
X = df_scaled.drop('target', axis=1)
y = df_scaled['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Save preprocessed data
df_scaled.to_csv("preprocessed_data.csv", index=False)
print("Preprocessed data saved successfully!")

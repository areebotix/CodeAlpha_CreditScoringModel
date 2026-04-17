# ================================
# Credit Risk Model Training Script
# ================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("../data/train.csv")

# -------------------------------
# Target and features
# -------------------------------
X = df.drop(columns=["target"])
y = df["target"]

# -------------------------------
# Identify column types
# -------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# -------------------------------
# Preprocessing
# -------------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# -------------------------------
# Model
# -------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "../models/credit_scoring_model.pkl")

print("✅ Model trained and saved successfully!")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. Load ML-ready data
# ---------------------------------------------------------

X_path = "data/processed/X.csv"
y_path = "data/processed/y.csv"

X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()


# ---------------------------------------------------------
# 2. Display dataset information
# ---------------------------------------------------------

print("=" * 60)
print("LOADED ML DATA")
print("=" * 60)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ---------------------------------------------------------
# 3. Split data into training and testing sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. Display split information
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training features: {X_train.shape}")
print(f"Testing features:  {X_test.shape}")

print(f"Training labels:   {y_train.shape}")
print(f"Testing labels:    {y_test.shape}")


# ---------------------------------------------------------
# 5. Check class distribution
# ---------------------------------------------------------

print("\nTraining placement distribution:")
print(y_train.value_counts())

print("\nTesting placement distribution:")
print(y_test.value_counts())


print("\nTrain/test split completed successfully!")
# ---------------------------------------------------------
# 6. Train XGBoost model
# ---------------------------------------------------------

from xgboost import XGBClassifier


print("\n" + "=" * 60)
print("TRAINING XGBOOST MODEL")
print("=" * 60)


model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)


model.fit(X_train, y_train)


print("XGBoost model training completed successfully!")
# ---------------------------------------------------------
# 7. Evaluate the model
# ---------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


# Make predictions on test data
y_pred = model.predict(X_test)

# Get placement probabilities
y_probability = model.predict_proba(X_test)[:, 1]


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

conf_matrix = confusion_matrix(y_test, y_pred)


print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


print("\nConfusion Matrix:")
print(conf_matrix)


print("\nModel evaluation completed successfully!")
# ---------------------------------------------------------
# 8. Save trained model
# ---------------------------------------------------------

model_path = "models/xgboost_placement_model.pkl"

joblib.dump(model, model_path)

print("\n" + "=" * 60)
print("MODEL SAVING")
print("=" * 60)

print(f"Model saved successfully!")
print(f"Model path: {model_path}")
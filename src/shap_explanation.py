import pandas as pd
import joblib
import shap

# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------

model_path = "models/xgboost_placement_model.pkl"

model = joblib.load(model_path)

print("=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

print("\nTrained XGBoost model loaded successfully!")


# ---------------------------------------------------------
# 2. Load test data
# ---------------------------------------------------------

X = pd.read_csv("data/processed/X.csv")

print(f"Feature data loaded: {X.shape}")


# ---------------------------------------------------------
# 3. Create SHAP explainer
# ---------------------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

print("\nSHAP values calculated successfully!")


# ---------------------------------------------------------
# 4. Display feature importance
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": abs(shap_values).mean(axis=0)
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 60)
print("SHAP FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)


# ---------------------------------------------------------
# 5. Display top 5 important features
# ---------------------------------------------------------

print("\nTop 5 factors influencing placement prediction:")

for index, row in feature_importance.head(5).iterrows():
    print(
        f"{row['feature']}: "
        f"{row['importance']:.4f}"
    )

print("\nSHAP analysis completed successfully!")
# ---------------------------------------------------------
# 6. Explain one individual student
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("INDIVIDUAL STUDENT EXPLANATION")
print("=" * 60)

# Select the first student from the dataset
student = X.iloc[[0]]

# Predict placement probability
placement_probability = model.predict_proba(student)[0][1]

# Calculate SHAP values for this student
student_shap = explainer.shap_values(student)[0]

# Create explanation table
student_explanation = pd.DataFrame({
    "feature": X.columns,
    "shap_value": student_shap,
    "feature_value": student.iloc[0].values
})

# Sort by absolute SHAP impact
student_explanation["absolute_impact"] = (
    student_explanation["shap_value"].abs()
)

student_explanation = student_explanation.sort_values(
    by="absolute_impact",
    ascending=False
)

print(f"\nPlacement probability: {placement_probability * 100:.2f}%")

print("\nTop factors affecting this student:")

print(
    student_explanation[
        ["feature", "feature_value", "shap_value"]
    ].head(10)
)

print("\nIndividual SHAP explanation completed successfully!")
import pandas as pd


# ---------------------------------------------------------
# 1. Load the raw dataset
# ---------------------------------------------------------

input_path = "data/raw/students.csv"

df = pd.read_csv(input_path)


# ---------------------------------------------------------
# 2. Basic information
# ---------------------------------------------------------

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())


# ---------------------------------------------------------
# 3. Check data types
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ---------------------------------------------------------
# 4. Check missing values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ---------------------------------------------------------
# 5. Check duplicate rows
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicates}")


# ---------------------------------------------------------
# 6. Check branch values
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BRANCH DISTRIBUTION")
print("=" * 60)

print(df["branch"].value_counts())


# ---------------------------------------------------------
# 7. Check placement distribution
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PLACEMENT DISTRIBUTION")
print("=" * 60)

print(df["placed"].value_counts())

print("\nPlacement percentage:")
print(df["placed"].value_counts(normalize=True) * 100)


# ---------------------------------------------------------
# 8. Numerical statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL STATISTICS")
print("=" * 60)

print(df.describe())


# ---------------------------------------------------------
# 9. Save a copy to processed folder
# ---------------------------------------------------------

output_path = "data/processed/students_clean.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Processed dataset saved to: {output_path}")
# ---------------------------------------------------------
# 10. Prepare features and target
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE PREPARATION")
print("=" * 60)

# Remove student_id because it is only an identifier
X = df.drop(columns=["student_id", "placed"])

# Target variable
y = df["placed"]

# One-hot encode the branch column
X = pd.get_dummies(
    X,
    columns=["branch"],
    dtype=int
)

print("\nFeatures after encoding:")
print(X.columns.tolist())

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of target values: {len(y)}")


# ---------------------------------------------------------
# 11. Save ML-ready data
# ---------------------------------------------------------

X_output_path = "data/processed/X.csv"
y_output_path = "data/processed/y.csv"

X.to_csv(X_output_path, index=False)
y.to_csv(y_output_path, index=False)

print("\nML-ready datasets saved:")
print(f"Features: {X_output_path}")
print(f"Target: {y_output_path}")
import numpy as np
import pandas as pd


# Make the generated dataset reproducible
np.random.seed(42)

# Number of students
N = 1000


# ---------------------------------------------------------
# 1. Basic student information
# ---------------------------------------------------------

student_id = [f"STU{i:04d}" for i in range(1, N + 1)]

branches = ["Computer", "IT", "AI&DS", "ENTC", "Mechanical", "Civil"]

branch = np.random.choice(
    branches,
    size=N,
    p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
)


# ---------------------------------------------------------
# 2. Academic performance
# ---------------------------------------------------------

cgpa = np.clip(
    np.random.normal(loc=7.6, scale=0.9, size=N),
    5.0,
    10.0
)

cgpa = np.round(cgpa, 2)


backlogs = np.random.choice(
    [0, 1, 2, 3, 4],
    size=N,
    p=[0.70, 0.15, 0.08, 0.05, 0.02]
)


# ---------------------------------------------------------
# 3. Skill-building activities
# ---------------------------------------------------------

certifications = np.random.poisson(lam=3, size=N)
certifications = np.clip(certifications, 0, 8)

projects = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    size=N,
    p=[0.05, 0.15, 0.25, 0.30, 0.18, 0.07]
)

dsa_problems = np.random.poisson(lam=70, size=N)
dsa_problems = np.clip(dsa_problems, 0, 250)


internship = np.random.choice(
    [0, 1],
    size=N,
    p=[0.55, 0.45]
)


# ---------------------------------------------------------
# 4. Soft skills
# ---------------------------------------------------------

communication_score = np.clip(
    np.random.normal(loc=68, scale=13, size=N),
    30,
    100
)

communication_score = np.round(communication_score, 1)


mock_interview_score = np.clip(
    np.random.normal(loc=65, scale=15, size=N),
    20,
    100
)

mock_interview_score = np.round(mock_interview_score, 1)


# ---------------------------------------------------------
# 5. Create a placement score
# ---------------------------------------------------------
# This is used only to generate realistic synthetic labels.
# The final ML model will learn the relationship itself.

placement_score = (
    0.30 * ((cgpa - 5) / 5) +
    0.15 * (1 - backlogs / 4) +
    0.08 * (certifications / 8) +
    0.12 * (projects / 5) +
    0.12 * (dsa_problems / 250) +
    0.08 * internship +
    0.08 * (communication_score / 100) +
    0.07 * (mock_interview_score / 100)
)


# Add a small amount of randomness
noise = np.random.normal(0, 0.08, N)

placement_score = placement_score + noise


# ---------------------------------------------------------
# 6. Convert score into placement label
# ---------------------------------------------------------

placed = (placement_score >= 0.55).astype(int)


# ---------------------------------------------------------
# 7. Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame({
    "student_id": student_id,
    "branch": branch,
    "cgpa": cgpa,
    "backlogs": backlogs,
    "certifications": certifications,
    "projects": projects,
    "dsa_problems": dsa_problems,
    "internship": internship,
    "communication_score": communication_score,
    "mock_interview_score": mock_interview_score,
    "placed": placed
})


# ---------------------------------------------------------
# 8. Save dataset
# ---------------------------------------------------------

output_path = "data/raw/students.csv"

df.to_csv(output_path, index=False)


# ---------------------------------------------------------
# 9. Display information
# ---------------------------------------------------------

print("Dataset generated successfully!")
print(f"Number of students: {len(df)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 students:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nPlacement distribution:")
print(df["placed"].value_counts())

print("\nPlacement percentage:")
print(df["placed"].mean() * 100)
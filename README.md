# 🎓 Campus Placement Predictor & Prep Recommender

An AI-powered student assessment system that estimates campus placement probability, explains the prediction using SHAP, identifies peer groups using K-Means clustering, and generates personalized placement preparation recommendations.

---

## 📌 Project Overview

Campus placement preparation can be difficult for students because they may not know which areas of their profile require improvement.

The **Campus Placement Predictor & Prep Recommender** analyzes academic, technical, internship and interview-related attributes of a student and provides:

- Placement probability estimation
- Prediction status
- SHAP-based explanation of the prediction
- Student peer-group clustering
- Personalized preparation recommendations
- Downloadable PDF assessment report
- Interactive Streamlit interface

The project is designed as an educational and demonstration system for understanding how machine learning can be applied to placement-readiness assessment.

---

## 🎯 Problem Statement

Students preparing for campus placements often have different strengths and weaknesses in areas such as:

- Academic performance
- Backlogs
- Certifications
- Projects
- Data Structures and Algorithms
- Internship experience
- Communication skills
- Mock interview performance

A student may know their individual scores but may not understand how these factors collectively affect a machine-learning model's placement prediction.

This project addresses this problem by combining **machine learning prediction, explainable AI, clustering and recommendation logic** into one application.

---

## 🚀 Key Features

### 1. 📈 Placement Prediction

The system uses **XGBoost Classification** to estimate the student's placement probability.

Input features include:

- CGPA
- Number of backlogs
- Number of certifications
- Number of projects
- DSA problems solved
- Internship experience
- Engineering branch
- Communication score
- Mock interview score

The system displays:

- Placement probability
- Prediction status

---

### 2. 🔍 SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

For every student, the system identifies features that contributed positively or negatively to the model's prediction.

Example:

```text
CGPA                 +1.24   Positive
Internship           +0.71   Positive
Projects             +0.42   Positive
Communication        -0.18   Negative
Backlogs             -0.35   Negative
```

A positive SHAP value means the feature pushed the model's prediction toward the "Placed" class.

A negative SHAP value means the feature pushed the model's prediction toward the "Not Placed" class.

SHAP values represent the model's contribution for a particular prediction. They should not be interpreted as causal effects.

---

### 3. 👥 Student Peer-Group Clustering

The system uses **K-Means Clustering** to group students according to similarities in their academic, technical and interview-related profiles.

The clustering process uses:

- StandardScaler
- K-Means
- PCA

Three clusters are generated in the current implementation.

The clusters are used to provide a simple peer-group description, such as:

```text
Students with no internship experience.
```

or

```text
Students with higher backlog levels.
```

The current synthetic dataset produces relatively weak cluster separation, so the clusters are treated as descriptive peer groups rather than definitive student categories.

---

### 4. 🎯 Personalized Preparation Plan

The recommendation engine identifies areas that may require additional preparation.

Possible recommendation areas include:

- CGPA
- Backlogs
- Certifications
- Projects
- DSA
- Internship
- Communication
- Mock Interview

Recommendations are assigned priorities such as:

- 🔴 High
- 🟡 Medium

Example:

```text
DSA — High Priority

Start with arrays, strings, searching and sorting,
then gradually move to medium-level problems.
```

---

### 5. 📄 PDF Assessment Report

The application can generate a downloadable PDF report containing:

- Student information
- Placement probability
- Prediction status
- Student cluster
- SHAP explanation
- Personalized preparation recommendations

The report is generated using **ReportLab**.

---

## 🧠 Machine Learning Workflow

```text
                 Student Profile
                       │
                       ▼
              Data Preprocessing
                       │
              ┌────────┴────────┐
              ▼                 ▼
       XGBoost Model       K-Means Model
              │                 │
              ▼                 ▼
      Placement Score      Peer Group
              │
              ▼
        SHAP Explanation
              │
              ▼
    Personalized Recommendations
              │
              ▼
        PDF Assessment Report
              │
              ▼
        Streamlit Dashboard
```

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Clustering & Visualization

- K-Means
- PCA
- Matplotlib

### Web Application

- Streamlit

### PDF Generation

- ReportLab

### Model Persistence

- Joblib
- Pickle

---

## 📊 Dataset

The current project uses a **synthetically generated dataset** containing 1,000 student records.

The dataset contains attributes such as:

| Feature | Description |
|---|---|
| student_id | Unique student identifier |
| branch | Engineering branch |
| cgpa | Student CGPA |
| backlogs | Number of backlogs |
| certifications | Number of certifications |
| projects | Number of projects |
| dsa_problems | Number of DSA problems solved |
| internship | Internship experience |
| communication_score | Communication score |
| mock_interview_score | Mock interview score |
| placed | Target placement label |

### Dataset Distribution

The generated dataset contains:

- Total students: **1,000**
- Placed: **504**
- Not placed: **496**

The dataset is intended for **educational and project demonstration purposes** and does not represent actual placement data from a college or company.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline:

1. Loads the raw student dataset.
2. Removes the `student_id` column from model features.
3. Separates the target variable `placed`.
4. Encodes the engineering branch using one-hot encoding.
5. Saves the cleaned dataset.
6. Saves the processed feature matrix.
7. Saves the target vector.

Processed files are stored in:

```text
data/processed/
```

---

## 📈 XGBoost Model

The placement prediction model uses:

```text
XGBClassifier
```

Current model configuration:

```python
n_estimators = 200
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

with stratified splitting.

---

## 📊 Model Evaluation

Current evaluation results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 69.00% |
| Precision | 69.31% |
| Recall | 69.31% |
| F1 Score | 69.31% |
| ROC-AUC | 76.67% |

### Confusion Matrix

```text
                 Predicted
                0       1

Actual 0       68      31
Actual 1       31      70
```

These metrics are based on the current synthetic dataset and should not be interpreted as real-world placement performance.

---

## 🔬 SHAP Analysis

SHAP is used to understand how the trained XGBoost model reaches individual predictions.

The current model's global feature importance shows that features such as:

- CGPA
- Internship
- Backlogs
- Projects
- DSA problems
- Communication score
- Mock interview score

have substantial influence on model predictions in the synthetic dataset.

SHAP provides model interpretability but does not establish that changing a feature will necessarily cause a real-world placement outcome.

---

## 👥 K-Means Clustering

The clustering model uses the following features:

```text
CGPA
Backlogs
Certifications
Projects
DSA Problems
Internship
Communication Score
Mock Interview Score
```

Before clustering, the features are standardized using:

```text
StandardScaler
```

The current implementation uses:

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

### PCA

PCA is used to visualize students in two dimensions.

The first two principal components currently explain approximately:

```text
27.97%
```

of the total variance.

Therefore, PCA is primarily used for visualization in this project rather than as a strong dimensionality-reduction representation.

---

## 🎯 Recommendation Engine

The recommendation engine uses rule-based thresholds to identify preparation gaps.

For example:

```text
CGPA < 7.0
→ High Priority CGPA recommendation
```

```text
Backlogs > 0
→ High Priority Backlog recommendation
```

```text
Projects < 2
→ High Priority Project recommendation
```

```text
Communication Score < 60
→ High Priority Communication recommendation
```

The recommendation engine combines these rules with the student's profile to generate a personalized preparation plan.

---

## 📂 Project Structure

```text
Campus_Placement_Predictor/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── students.csv
│   │
│   └── processed/
│       ├── students_clean.csv
│       ├── X.csv
│       └── y.csv
│
├── models/
│   ├── xgboost_placement_model.pkl
│   ├── clustering_scaler.pkl
│   └── kmeans_clustering_model.pkl
│
├── reports/
│   └── student_clusters_pca.png
│
├── src/
│   ├── generate_dataset.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── shap_explanation.py
│   ├── clustering.py
│   ├── recommender.py
│   └── pdf_report.py
│
└── venv/
```

The `venv/` directory is local to the development environment and is excluded from Git using `.gitignore`.

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project

```bash
cd Campus_Placement_Predictor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔄 Rebuilding the Machine Learning Pipeline

If the dataset needs to be regenerated, run:

```bash
python src/generate_dataset.py
```

Then preprocess the data:

```bash
python src/data_preprocessing.py
```

Train the placement model:

```bash
python src/train_model.py
```

Run SHAP analysis:

```bash
python src/shap_explanation.py
```

Run clustering:

```bash
python src/clustering.py
```

After the models are generated, start the application:

```bash
streamlit run app.py
```

---

## ⚠️ Limitations

The current project has several limitations.

### 1. Synthetic Dataset

The current dataset is artificially generated and does not represent real college placement data.

Therefore, the model's predictions should not be treated as actual placement probabilities for real students.

### 2. Limited Real-World Generalization

A model trained on synthetic data may not generalize to actual placement outcomes because real placement decisions can depend on many additional factors.

### 3. Clustering Separation

The current K-Means clusters have relatively low separation according to the silhouette analysis.

Therefore, clusters should be interpreted as approximate peer groups rather than strict categories.

### 4. Rule-Based Recommendations

The recommendation engine currently uses predefined thresholds rather than a learned recommendation model.

### 5. Placement Probability

The displayed percentage is the model's estimated probability for the supplied profile. It is not a guarantee of placement.

---

## 🔮 Future Scope

Possible improvements include:

- Train the model using a larger real-world dataset.
- Add more placement-related features.
- Improve class calibration and probability reliability.
- Compare XGBoost with Random Forest, Logistic Regression and other classifiers.
- Improve cluster separation using better feature engineering.
- Add interactive PCA cluster visualization.
- Develop a stronger recommendation engine.
- Add resume analysis.
- Add job-role recommendations.
- Add skill-gap analysis.
- Add company-specific eligibility simulation using publicly available criteria.
- Add authentication and student profiles.
- Deploy the application online.

---

## 🎓 Learning Outcomes

This project demonstrates practical understanding of:

- Data preprocessing
- Feature engineering
- Classification
- XGBoost
- Model evaluation
- SHAP explainability
- K-Means clustering
- PCA
- Rule-based recommendation systems
- Model persistence
- Streamlit application development
- PDF report generation
- End-to-end machine learning workflow

---

## 👩‍💻 Project Purpose

This project was developed as an academic and portfolio project to demonstrate the integration of **Machine Learning, Explainable AI, Clustering and Recommendation Systems** into an interactive application.

It is intended for educational and demonstration purposes.

---

## 📜 Disclaimer

This application is a machine-learning project for educational purposes.

The placement probability generated by the application is not a guarantee of actual placement. The current model is trained on synthetic data and should not be used to make real-world employment decisions.

---

## ⭐ Technologies

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
K-Means
PCA
Matplotlib
Streamlit
ReportLab
Joblib
```
## Git & GitHub Learning

This project is also being used to learn Git and GitHub workflows.

## Dashboard Feature

Dashboard development is being tested on a separate Git branch.

## Current Development

This section is being updated as part of the feature development workflow.
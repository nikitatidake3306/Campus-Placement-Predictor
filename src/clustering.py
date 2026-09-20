import streamlit as st
import pandas as pd
import joblib
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

xgb_model = joblib.load(
    "models/xgboost_placement_model.pkl"
)

clustering_scaler = joblib.load(
    "models/clustering_scaler.pkl"
)

kmeans_model = joblib.load(
    "models/kmeans_clustering_model.pkl"
)

shap_explainer = shap.TreeExplainer(
    xgb_model
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🎓 Campus Placement Predictor & Prep Recommender"
)

st.write(
    "Predict placement probability, understand the prediction "
    "using SHAP, and identify the student's peer group using K-Means."
)


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.header("Student Information")


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    projects = st.number_input(
        "Projects",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    dsa_problems = st.number_input(
        "DSA Problems Solved",
        min_value=0,
        max_value=500,
        value=50,
        step=1
    )

    internship = st.selectbox(
        "Internship",
        ["No", "Yes"]
    )

    communication_score = st.slider(
        "Communication Score",
        min_value=0,
        max_value=100,
        value=65
    )

    mock_interview_score = st.slider(
        "Mock Interview Score",
        min_value=0,
        max_value=100,
        value=60
    )


# ============================================================
# BRANCH
# ============================================================

branch = st.selectbox(
    "Branch",
    [
        "Computer",
        "IT",
        "AI&DS",
        "ENTC",
        "Mechanical",
        "Civil"
    ]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict Placement"):

    # --------------------------------------------------------
    # Convert internship to numerical value
    # --------------------------------------------------------

    internship_value = 1 if internship == "Yes" else 0


    # --------------------------------------------------------
    # Create student data
    # --------------------------------------------------------

    student_data = pd.DataFrame({
        "cgpa": [cgpa],
        "backlogs": [backlogs],
        "certifications": [certifications],
        "projects": [projects],
        "dsa_problems": [dsa_problems],
        "internship": [internship_value],
        "communication_score": [communication_score],
        "mock_interview_score": [mock_interview_score]
    })


    # ========================================================
    # BRANCH ONE-HOT ENCODING
    # ========================================================

    branch_names = [
        "AI&DS",
        "Civil",
        "Computer",
        "ENTC",
        "IT",
        "Mechanical"
    ]


    for branch_name in branch_names:

        column_name = f"branch_{branch_name}"

        if branch == branch_name:
            student_data[column_name] = 1
        else:
            student_data[column_name] = 0


    # ========================================================
    # XGBOOST FEATURE ORDER
    # ========================================================

    feature_columns = [
        "cgpa",
        "backlogs",
        "certifications",
        "projects",
        "dsa_problems",
        "internship",
        "communication_score",
        "mock_interview_score",
        "branch_AI&DS",
        "branch_Civil",
        "branch_Computer",
        "branch_ENTC",
        "branch_IT",
        "branch_Mechanical"
    ]


    student_data = student_data[
        feature_columns
    ]


    # ========================================================
    # XGBOOST PREDICTION
    # ========================================================

    probability = xgb_model.predict_proba(
        student_data
    )[0][1]


    prediction = xgb_model.predict(
        student_data
    )[0]


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    shap_values = shap_explainer.shap_values(
        student_data
    )


    student_shap = shap_values[0]


    shap_explanation = pd.DataFrame({
        "Feature": feature_columns,
        "SHAP Value": student_shap
    })


    shap_explanation["Absolute Impact"] = (
        shap_explanation["SHAP Value"].abs()
    )


    shap_explanation = shap_explanation.sort_values(
        by="Absolute Impact",
        ascending=False
    )


    # ========================================================
    # K-MEANS CLUSTERING
    # ========================================================

    clustering_features = [
        "cgpa",
        "backlogs",
        "certifications",
        "projects",
        "dsa_problems",
        "internship",
        "communication_score",
        "mock_interview_score"
    ]


    clustering_data = student_data[
        clustering_features
    ]


    # Standardize using the SAME scaler used during training

    clustering_scaled = clustering_scaler.transform(
        clustering_data
    )


    # Predict cluster

    student_cluster = kmeans_model.predict(
        clustering_scaled
    )[0]


    # ========================================================
    # PLACEMENT RESULT
    # ========================================================

    st.header("📊 Placement Prediction")


    st.metric(
        "Placement Probability",
        f"{probability * 100:.2f}%"
    )


    if prediction == 1:

        st.success(
            "The model predicts that the student is likely to be placed."
        )

    else:

        st.warning(
            "The model predicts that the student may need more preparation."
        )


    # ========================================================
    # SHAP RESULT
    # ========================================================

    st.header(
        "🔍 Why did the model make this prediction?"
    )


    st.write(
        "SHAP explains how each feature influenced the "
        "placement prediction."
    )


    display_shap = shap_explanation[
        ["Feature", "SHAP Value"]
    ].head(10)


    st.dataframe(
        display_shap,
        width="stretch"
    )


    st.subheader(
        "How to read SHAP values"
    )


    st.write(
        "Positive SHAP values push the prediction toward "
        "Placed, while negative SHAP values push the prediction "
        "toward Not Placed."
    )


    # ========================================================
    # STUDENT CLUSTER
    # ========================================================

    st.header(
        "👥 Student Peer Group"
    )


    st.metric(
        "Student Cluster",
        f"Cluster {student_cluster}"
    )


    # --------------------------------------------------------
    # Cluster descriptions based on our current dataset
    # --------------------------------------------------------

    cluster_descriptions = {

        0: (
            "Higher-backlog student group. "
            "Students in this group generally have more active "
            "backlogs and may need stronger academic improvement."
        ),

        1: (
            "No-internship student group. "
            "Students in this group generally have no internship "
            "experience and can benefit from practical project or "
            "internship experience."
        ),

        2: (
            "Internship-experienced student group. "
            "Students in this group generally have internship "
            "experience and can focus on strengthening advanced "
            "interview preparation."
        )
    }


    st.info(
        cluster_descriptions.get(
            int(student_cluster),
            "Student belongs to a peer group with similar preparation characteristics."
        )
    )


    st.write(
        "The cluster is identified using K-Means based on academic "
        "performance, skills, internship experience and interview "
        "readiness."
    )
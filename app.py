# ============================================================
# CAMPUS PLACEMENT PREDICTOR & PREP RECOMMENDER
# ============================================================

import os
import sys
import joblib
import pandas as pd
import streamlit as st
import shap


# ============================================================
# ADD SRC FOLDER TO PYTHON PATH
# ============================================================

SRC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src"
)

sys.path.insert(0, SRC_PATH)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from recommender import generate_recommendations
from pdf_report import generate_pdf_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Campus Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    ">
        🎓 Campus Placement Predictor
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    ">
        AI-powered placement prediction, SHAP explainability,
        student clustering and personalized preparation
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    model = joblib.load(
        "models/xgboost_placement_model.pkl"
    )

    scaler = joblib.load(
        "models/clustering_scaler.pkl"
    )

    kmeans = joblib.load(
        "models/kmeans_clustering_model.pkl"
    )

    return model, scaler, kmeans


model, scaler, kmeans = load_models()


# ============================================================
# STUDENT PROFILE
# ============================================================

st.header("📝 Student Profile")

st.write(
    "Enter the student's academic, technical and "
    "interview-related information."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🎓 Academic & Technical Profile")

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
        step=0.01
    )

    backlogs = st.number_input(
        "Number of Backlogs",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    certifications = st.number_input(
        "Number of Certifications",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    dsa_problems = st.number_input(
        "DSA Problems Solved",
        min_value=0,
        max_value=500,
        value=50,
        step=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    internship = st.selectbox(
        "Internship Experience",
        ["No", "Yes"]
    )

    branch = st.selectbox(
        "Engineering Branch",
        [
            "Computer",
            "IT",
            "AI&DS",
            "ENTC",
            "Mechanical",
            "Civil"
        ]
    )

    communication_score = st.slider(
        "Communication Score",
        0,
        100,
        65
    )

    mock_interview_score = st.slider(
        "Mock Interview Score",
        0,
        100,
        60
    )


# ============================================================
# GENERATE ASSESSMENT
# ============================================================

st.markdown("")

generate_button = st.button(
    "🚀 Generate Assessment",
    use_container_width=True
)


# ============================================================
# GENERATE RESULTS
# ============================================================

if generate_button:

    # ========================================================
    # INTERNSHIP VALUE
    # ========================================================

    internship_value = (
        1 if internship == "Yes" else 0
    )


    # ========================================================
    # STUDENT PROFILE
    # ========================================================

    student = {

        "cgpa": cgpa,

        "backlogs": backlogs,

        "certifications": certifications,

        "projects": projects,

        "dsa_problems": dsa_problems,

        "internship": internship_value,

        "communication_score": communication_score,

        "mock_interview_score": mock_interview_score,

        "branch": branch
    }


    # ========================================================
    # MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame(
        {
            "cgpa": [cgpa],

            "backlogs": [backlogs],

            "certifications": [certifications],

            "projects": [projects],

            "dsa_problems": [dsa_problems],

            "internship": [internship_value],

            "communication_score": [
                communication_score
            ],

            "mock_interview_score": [
                mock_interview_score
            ],

            "branch": [branch]
        }
    )


    # ========================================================
    # ONE-HOT ENCODE BRANCH
    # ========================================================

    input_data = pd.get_dummies(
        input_data,
        columns=["branch"],
        dtype=int
    )


    # ========================================================
    # MODEL FEATURES
    # ========================================================

    expected_features = [
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


    for feature in expected_features:

        if feature not in input_data.columns:

            input_data[feature] = 0


    input_data = input_data[
        expected_features
    ]


    # ========================================================
    # PLACEMENT PREDICTION
    # ========================================================

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]


    probability_percentage = (
        probability * 100
    )


    # ========================================================
    # STATUS
    # ========================================================

    if prediction == 1:

        prediction_status = "Likely to be Placed"

    else:

        prediction_status = "Needs More Preparation"


    # ========================================================
    # CLUSTERING DATA
    # ========================================================

    clustering_input = pd.DataFrame(
        {
            "cgpa": [cgpa],

            "backlogs": [backlogs],

            "certifications": [certifications],

            "projects": [projects],

            "dsa_problems": [dsa_problems],

            "internship": [internship_value],

            "communication_score": [
                communication_score
            ],

            "mock_interview_score": [
                mock_interview_score
            ]
        }
    )


    clustering_scaled = scaler.transform(
        clustering_input
    )


    cluster = kmeans.predict(
        clustering_scaled
    )[0]


    cluster_name = f"Cluster {cluster}"


    # ========================================================
    # CLUSTER DESCRIPTION
    # ========================================================

    cluster_descriptions = {

        0:
        "Students with higher backlog levels.",

        1:
        "Students with no internship experience.",

        2:
        "Students with internship experience."
    }


    cluster_description = cluster_descriptions.get(
        cluster,
        "Students with similar academic and placement preparation profiles."
    )


    # ========================================================
    # SHAP
    # ========================================================

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        input_data
    )


    # Handle SHAP output

    if isinstance(shap_values, list):

        shap_row = shap_values[1][0]

    else:

        if len(shap_values.shape) == 3:

            shap_row = shap_values[0, :, 1]

        else:

            shap_row = shap_values[0]


    # ========================================================
    # SHAP DATAFRAME
    # ========================================================

    shap_explanation = pd.DataFrame(
        {
            "Feature": input_data.columns,

            "SHAP Value": shap_row
        }
    )


    shap_explanation["Absolute SHAP"] = (
        shap_explanation["SHAP Value"].abs()
    )


    shap_explanation = (
        shap_explanation
        .sort_values(
            "Absolute SHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = generate_recommendations(
        student
    )


    # ========================================================
    # PLACEMENT PREDICTION
    # ========================================================

    st.header("📊 Placement Prediction")


    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "📈 Placement Probability",
            f"{probability_percentage:.2f}%"
        )


    with result2:

        st.metric(
            "🎯 Prediction Status",
            prediction_status
        )


    with result3:

        st.metric(
            "👥 Student Peer Group",
            cluster_name
        )


    # ========================================================
    # PROBABILITY INDICATOR
    # ========================================================

    st.subheader(
        "📊 Placement Probability Indicator"
    )


    st.progress(
        float(probability)
    )


    if probability < 0.40:

        st.warning(
            "The model indicates a lower placement probability. "
            "Focus on the recommended preparation areas."
        )

    elif probability < 0.70:

        st.info(
            "The model indicates moderate placement readiness. "
            "Continue improving your weaker areas."
        )

    else:

        st.success(
            "The model indicates strong placement readiness. "
            "Continue practicing and maintain your preparation."
        )


    # ========================================================
    # PEER GROUP
    # ========================================================

    st.subheader(
        "👥 Student Peer Group"
    )


    st.info(
        f"**{cluster_name}**\n\n"
        f"{cluster_description}"
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.header(
        "🔍 Why did the model make this prediction?"
    )


    st.write(
        "SHAP explains how each feature influenced the "
        "placement prediction."
    )


    # ========================================================
    # TOP 10 SHAP FEATURES
    # ========================================================

    display_shap = shap_explanation[
        ["Feature", "SHAP Value"]
    ].head(10).copy()


    display_shap["SHAP Value"] = (
        display_shap["SHAP Value"].round(4)
    )


    display_shap["Impact"] = (
        display_shap["SHAP Value"]
        .apply(
            lambda x:
            "🟢 Positive"
            if x > 0
            else "🔴 Negative"
        )
    )


    # ========================================================
    # SHAP TABLE
    # ========================================================

    st.table(
        display_shap
    )


    # ========================================================
    # SHAP INTERPRETATION
    # ========================================================

    st.subheader(
        "📌 How to interpret the SHAP values"
    )


    st.markdown(
        """
        **🟢 Positive SHAP value:**  
        This feature pushed the model's prediction toward
        **Placed**.

        **🔴 Negative SHAP value:**  
        This feature pushed the model's prediction toward
        **Not Placed**.

        The larger the absolute SHAP value, the stronger the
        feature's influence on this particular prediction.
        """
    )


    # ========================================================
    # PERSONALIZED PREPARATION PLAN
    # ========================================================

    st.header(
        "🎯 Personalized Preparation Plan"
    )


    st.write(
        "Based on your profile, the system recommends "
        "the following areas for placement preparation."
    )


    if len(recommendations) == 0:

        st.success(
            "Excellent! No major preparation gaps were detected."
        )

    else:

        for item in recommendations:

            area = item["area"]

            priority = item["priority"]

            recommendation = item["recommendation"]


            if priority == "High":

                st.error(
                    f"🔴 {area} — High Priority\n\n"
                    f"{recommendation}"
                )

            elif priority == "Medium":

                st.warning(
                    f"🟡 {area} — Medium Priority\n\n"
                    f"{recommendation}"
                )

            else:

                st.info(
                    f"🟢 {area} — Low Priority\n\n"
                    f"{recommendation}"
                )


    # ========================================================
    # PDF REPORT
    # ========================================================

    st.header(
        "📄 Placement Assessment Report"
    )


    st.write(
        "Generate a downloadable PDF containing the "
        "prediction, SHAP explanation and personalized "
        "preparation plan."
    )


    os.makedirs(
        "reports",
        exist_ok=True
    )


    pdf_path = (
        "reports/student_placement_report.pdf"
    )


    try:

        generate_pdf_report(

            student=student,

            placement_probability=(
                probability_percentage
            ),

            prediction=prediction_status,

            cluster=cluster_name,

            shap_explanation=(
                shap_explanation[
                    ["Feature", "SHAP Value"]
                ].head(10)
            ),

            recommendations=recommendations,

            output_path=pdf_path
        )


        with open(
            pdf_path,
            "rb"
        ) as pdf_file:

            st.download_button(

                label=(
                    "📥 Download Placement Assessment Report"
                ),

                data=pdf_file,

                file_name=(
                    "student_placement_report.pdf"
                ),

                mime="application/pdf",

                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Could not generate PDF report: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">

    <b>Campus Placement Predictor & Prep Recommender</b>

    <br><br>

    Built using Python, XGBoost, SHAP,
    K-Means, PCA and Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
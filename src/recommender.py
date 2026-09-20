# ============================================================
# PERSONALIZED PREPARATION RECOMMENDER
# ============================================================


def generate_recommendations(student):
    """
    Generate personalized placement preparation
    recommendations based on student profile.
    """

    recommendations = []


    # ========================================================
    # CGPA
    # ========================================================

    if student["cgpa"] < 7.0:

        recommendations.append({
            "area": "CGPA",
            "priority": "High",
            "recommendation":
                "Focus on improving academic performance and "
                "maintain a strong semester GPA."
        })

    elif student["cgpa"] < 8.0:

        recommendations.append({
            "area": "CGPA",
            "priority": "Medium",
            "recommendation":
                "Maintain your academic performance and try to "
                "improve your CGPA above 8.0."
        })


    # ========================================================
    # BACKLOGS
    # ========================================================

    if student["backlogs"] > 0:

        recommendations.append({
            "area": "Backlogs",
            "priority": "High",
            "recommendation":
                "Clear all active backlogs because some companies "
                "have strict eligibility criteria."
        })


    # ========================================================
    # CERTIFICATIONS
    # ========================================================

    if student["certifications"] < 2:

        recommendations.append({
            "area": "Certifications",
            "priority": "Medium",
            "recommendation":
                "Complete relevant certifications in Python, SQL, "
                "Data Analytics or Machine Learning."
        })


    # ========================================================
    # PROJECTS
    # ========================================================

    if student["projects"] < 2:

        recommendations.append({
            "area": "Projects",
            "priority": "High",
            "recommendation":
                "Build at least one strong industry-oriented project "
                "and be ready to explain it in interviews."
        })


    # ========================================================
    # DSA
    # ========================================================

    if student["dsa_problems"] < 50:

        recommendations.append({
            "area": "DSA",
            "priority": "High",
            "recommendation":
                "Start with arrays, strings, searching and sorting, "
                "then gradually move to medium-level problems."
        })

    elif student["dsa_problems"] < 100:

        recommendations.append({
            "area": "DSA",
            "priority": "Medium",
            "recommendation":
                "Continue solving DSA problems regularly and focus "
                "on improving problem-solving speed."
        })


    # ========================================================
    # INTERNSHIP
    # ========================================================

    if student["internship"] == 0:

        recommendations.append({
            "area": "Internship",
            "priority": "Medium",
            "recommendation":
                "Gain practical experience through internships, "
                "live projects or relevant industry work."
        })


    # ========================================================
    # COMMUNICATION
    # ========================================================

    if student["communication_score"] < 60:

        recommendations.append({
            "area": "Communication",
            "priority": "High",
            "recommendation":
                "Practice English speaking daily, participate in "
                "group discussions and improve interview communication."
        })

    elif student["communication_score"] < 75:

        recommendations.append({
            "area": "Communication",
            "priority": "Medium",
            "recommendation":
                "Continue daily speaking practice and focus on "
                "confidence, vocabulary and pronunciation."
        })


    # ========================================================
    # MOCK INTERVIEW
    # ========================================================

    if student["mock_interview_score"] < 60:

        recommendations.append({
            "area": "Mock Interview",
            "priority": "High",
            "recommendation":
                "Take regular mock interviews and practice "
                "technical, HR and project-based questions."
        })

    elif student["mock_interview_score"] < 75:

        recommendations.append({
            "area": "Mock Interview",
            "priority": "Medium",
            "recommendation":
                "Practice mock interviews regularly and improve "
                "your answers and confidence."
        })


    # ========================================================
    # RETURN RECOMMENDATIONS
    # ========================================================

    return recommendations


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PERSONALIZED PREPARATION RECOMMENDER")
    print("=" * 60)


    # --------------------------------------------------------
    # Test student
    # --------------------------------------------------------

    student = {

        "cgpa": 7.2,

        "backlogs": 1,

        "certifications": 1,

        "projects": 1,

        "dsa_problems": 40,

        "internship": 0,

        "communication_score": 50,

        "mock_interview_score": 55

    }


    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = generate_recommendations(
        student
    )


    # --------------------------------------------------------
    # Display student profile
    # --------------------------------------------------------

    print("\nStudent Profile:")
    print(student)


    # --------------------------------------------------------
    # Display preparation plan
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PERSONALIZED PREPARATION PLAN")
    print("=" * 60)


    for i, item in enumerate(
        recommendations,
        start=1
    ):

        print(f"\n{i}. {item['area']}")

        print(
            f"   Priority: {item['priority']}"
        )

        print(
            f"   Recommendation: "
            f"{item['recommendation']}"
        )


    print("\n" + "=" * 60)
    print("RECOMMENDER TEST COMPLETED")
    print("=" * 60)
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


def generate_pdf_report(
    student,
    placement_probability,
    prediction,
    cluster,
    shap_explanation,
    recommendations,
    output_path
):

    # ============================================================
    # PDF DOCUMENT
    # ============================================================

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=12
    )


    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8
    )


    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12
    )


    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10
    )


    story = []


    # ============================================================
    # TITLE
    # ============================================================

    story.append(
        Paragraph(
            "Campus Placement Predictor & Prep Recommender",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Placement Assessment Report",
            ParagraphStyle(
                "Subtitle",
                parent=styles["BodyText"],
                alignment=TA_CENTER,
                fontSize=11,
                leading=14
            )
        )
    )

    story.append(Spacer(1, 15))


    # ============================================================
    # STUDENT INFORMATION
    # ============================================================

    story.append(
        Paragraph(
            "1. Student Profile",
            heading_style
        )
    )


    student_data = [
        [
            Paragraph("<b>Feature</b>", small_style),
            Paragraph("<b>Value</b>", small_style)
        ],
        [
            Paragraph("CGPA", small_style),
            Paragraph(str(student["cgpa"]), small_style)
        ],
        [
            Paragraph("Backlogs", small_style),
            Paragraph(str(student["backlogs"]), small_style)
        ],
        [
            Paragraph("Certifications", small_style),
            Paragraph(str(student["certifications"]), small_style)
        ],
        [
            Paragraph("Projects", small_style),
            Paragraph(str(student["projects"]), small_style)
        ],
        [
            Paragraph("DSA Problems Solved", small_style),
            Paragraph(str(student["dsa_problems"]), small_style)
        ],
        [
            Paragraph("Internship", small_style),
            Paragraph(
                "Yes" if student["internship"] == 1 else "No",
                small_style
            )
        ],
        [
            Paragraph("Communication Score", small_style),
            Paragraph(str(student["communication_score"]), small_style)
        ],
        [
            Paragraph("Mock Interview Score", small_style),
            Paragraph(str(student["mock_interview_score"]), small_style)
        ],
    ]


    student_table = Table(
        student_data,
        colWidths=[230, 290]
    )


    student_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )


    story.append(student_table)
    story.append(Spacer(1, 15))


    # ============================================================
    # PLACEMENT PREDICTION
    # ============================================================

    story.append(
        Paragraph(
            "2. Placement Prediction",
            heading_style
        )
    )


    prediction_text = (
        "Likely to be Placed"
        if prediction == 1
        else
        "Needs More Preparation"
    )


    prediction_data = [
        [
            Paragraph("<b>Placement Probability</b>", small_style),
            Paragraph(
                f"<b>{placement_probability:.2f}%</b>",
                small_style
            )
        ],
        [
            Paragraph("<b>Prediction Status</b>", small_style),
            Paragraph(prediction_text, small_style)
        ],
        [
            Paragraph("<b>Student Cluster</b>", small_style),
            Paragraph(f"Cluster {cluster}", small_style)
        ]
    ]


    prediction_table = Table(
        prediction_data,
        colWidths=[230, 290]
    )


    prediction_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )


    story.append(prediction_table)
    story.append(Spacer(1, 15))


    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    story.append(
        Paragraph(
            "3. SHAP Explanation",
            heading_style
        )
    )


    story.append(
        Paragraph(
            "SHAP values explain how individual features influenced "
            "the model's placement prediction. Positive values push "
            "the prediction toward Placed, while negative values push "
            "the prediction toward Not Placed.",
            normal_style
        )
    )


    story.append(Spacer(1, 8))


    # ------------------------------------------------------------
    # Convert DataFrame into rows safely
    # ------------------------------------------------------------

    shap_table_data = [
        [
            Paragraph("<b>Feature</b>", small_style),
            Paragraph("<b>SHAP Value</b>", small_style),
            Paragraph("<b>Impact</b>", small_style)
        ]
    ]


    # Take top 10 SHAP features
    shap_rows = shap_explanation.head(10)


    for _, row in shap_rows.iterrows():

        feature = str(row["Feature"])
        shap_value = float(row["SHAP Value"])


        if shap_value >= 0:
            impact = "Positive"
        else:
            impact = "Negative"


        shap_table_data.append(
            [
                Paragraph(feature, small_style),
                Paragraph(
                    f"{shap_value:.4f}",
                    small_style
                ),
                Paragraph(
                    impact,
                    small_style
                )
            ]
        )


    shap_table = Table(
        shap_table_data,
        colWidths=[250, 120, 150],
        repeatRows=1
    )


    shap_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )


    story.append(shap_table)
    story.append(Spacer(1, 15))


    # ============================================================
    # PERSONALIZED RECOMMENDATIONS
    # ============================================================

    story.append(
        Paragraph(
            "4. Personalized Preparation Plan",
            heading_style
        )
    )


    if len(recommendations) == 0:

        story.append(
            Paragraph(
                "No major preparation gaps were detected.",
                normal_style
            )
        )

    else:

        recommendation_data = [
            [
                Paragraph("<b>Area</b>", small_style),
                Paragraph("<b>Priority</b>", small_style),
                Paragraph("<b>Recommendation</b>", small_style)
            ]
        ]


        for item in recommendations:

            recommendation_data.append(
                [
                    Paragraph(
                        str(item["area"]),
                        small_style
                    ),
                    Paragraph(
                        str(item["priority"]),
                        small_style
                    ),
                    Paragraph(
                        str(item["recommendation"]),
                        small_style
                    )
                ]
            )


        recommendation_table = Table(
            recommendation_data,
            colWidths=[90, 75, 355],
            repeatRows=1
        )


        recommendation_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )


        story.append(recommendation_table)


    story.append(Spacer(1, 20))


    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated using a "
            "machine-learning model trained on synthetic student data. "
            "The placement probability is a model-generated estimate "
            "and should not be interpreted as a guaranteed placement "
            "outcome.",
            small_style
        )
    )


    story.append(Spacer(1, 10))


    story.append(
        Paragraph(
            "Built using Python, XGBoost, SHAP, K-Means, PCA and Streamlit.",
            small_style
        )
    )


    # ============================================================
    # BUILD PDF
    # ============================================================

    doc.build(story)
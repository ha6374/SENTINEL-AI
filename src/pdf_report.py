from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def generate_pdf_report(
    filename,
    prediction,
    confidence,
    real_probability,
    fake_probability,
    duration,
    sample_rate=16000,
):
    """Generate a downloadable PDF report for an audio prediction."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="SENTINEL AI Audio Analysis Report",
        author="SENTINEL AI",
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="SentinelTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#147D45"),
        spaceAfter=5 * mm,
    ))

    styles.add(ParagraphStyle(
        name="SentinelSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=7 * mm,
    ))

    styles.add(ParagraphStyle(
        name="SentinelHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#147D45"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    ))

    styles.add(ParagraphStyle(
        name="SentinelBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#333333"),
    ))

    safe_filename = escape(str(filename))
    prediction = str(prediction)

    if prediction not in ("Real", "Fake"):
        raise ValueError("Prediction must be Real or Fake.")

    confidence = float(confidence)
    real_probability = float(real_probability)
    fake_probability = float(fake_probability)
    duration = float(duration)
    sample_rate = int(sample_rate)

    for value in (confidence, real_probability, fake_probability):
        if not 0 <= value <= 1:
            raise ValueError("Probabilities and confidence must be between 0 and 1.")

    if duration < 0 or sample_rate <= 0:
        raise ValueError("Invalid audio duration or sample rate.")

    generated_at = datetime.now().astimezone().strftime(
        "%d %B %Y, %I:%M:%S %p %Z"
    )

    report_id = datetime.now().strftime("SA-%Y%m%d-%H%M%S")

    story = []

    story.append(Paragraph("SENTINEL AI", styles["SentinelTitle"]))
    story.append(Paragraph(
        "DEEPFAKE AUDIO DETECTION · ANALYSIS REPORT",
        styles["SentinelSubtitle"],
    ))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#147D45"),
        spaceAfter=6 * mm,
    ))

    story.append(Paragraph("Analysis Overview", styles["SentinelHeading"]))

    overview = [
        ["Report ID", report_id],
        ["Generated", generated_at],
        ["Audio filename", safe_filename],
        ["Audio duration", f"{duration:.2f} seconds"],
        ["Sample rate", f"{sample_rate:,} Hz"],
        ["Model", "SENTINEL AI trained CNN"],
    ]

    table = Table(overview, colWidths=[48 * mm, 115 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF5EE")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#14532D")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5E2D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(table)
    story.append(Paragraph("Classification Result", styles["SentinelHeading"]))

    result_color = "#147D45" if prediction == "Real" else "#B42318"

    result_table = Table([
        [Paragraph(
            f'<font color="{result_color}"><b>{escape(prediction.upper())} AUDIO</b></font>',
            styles["Heading1"],
        )],
        [Paragraph(
            f"Model confidence: <b>{confidence:.2%}</b>",
            styles["SentinelBody"],
        )],
    ], colWidths=[163 * mm])

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7F4")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(result_color)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    story.append(result_table)

    story.append(Paragraph("Class Probabilities", styles["SentinelHeading"]))

    probabilities = [
        ["Class", "Probability"],
        ["Real audio", f"{real_probability:.2%}"],
        ["Fake audio", f"{fake_probability:.2%}"],
    ]

    probability_table = Table(
        probabilities,
        colWidths=[105 * mm, 58 * mm],
    )

    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#147D45")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5E2D8")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(probability_table)

    story.append(Paragraph("Important Limitations", styles["SentinelHeading"]))
    story.append(Paragraph(
        "This report contains an automated model prediction, not proof that "
        "an audio recording is authentic or manipulated. Model confidence "
        "is not a guarantee of correctness. Performance may vary on audio "
        "from unfamiliar speakers, recording devices, languages, and "
        "manipulation methods. Use independent verification for important decisions.",
        styles["SentinelBody"],
    ))

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(
        width="100%",
        thickness=0.5,
        color=colors.lightgrey,
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "Generated by SENTINEL AI · Deepfake Audio Detector",
        styles["SentinelSubtitle"],
    ))

    doc.build(story)

    buffer.seek(0)
    return buffer.getvalue()

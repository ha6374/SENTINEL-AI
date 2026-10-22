import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def render_performance_dashboard():
    st.title("📊 Model Performance Dashboard")
    st.caption("SENTINEL AI | Deepfake Audio Detection")

    st.info(
        "The following metrics are from the ASVspoof2019 LA development "
        "dataset evaluation. They are development-set results, not "
        "independent final test results."
    )

    # Evaluation metrics
    accuracy = 98.30
    real_precision = 95.27
    real_recall = 87.76
    real_f1 = 91.36
    fake_precision = 98.61
    fake_recall = 99.50
    fake_f1 = 99.06

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Real Recall", f"{real_recall:.2f}%")
    col3.metric("Fake Recall", f"{fake_recall:.2f}%")

    st.divider()
    st.subheader("📈 Classification Metrics")

    metric_names = [
        "Precision",
        "Recall",
        "F1 Score"
    ]

    real_values = [
        real_precision,
        real_recall,
        real_f1
    ]

    fake_values = [
        fake_precision,
        fake_recall,
        fake_f1
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(x - width / 2, real_values, width, label="Real")
    ax.bar(x + width / 2, fake_values, width, label="Fake")

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("🔍 Confusion Matrix")

    st.write(
        "This matrix shows the number of correctly and incorrectly "
        "classified audio samples."
    )

    confusion_matrix = np.array([
        [2236, 312],
        [111, 22185]
    ])

    fig, ax = plt.subplots(figsize=(6, 4))

    image = ax.imshow(confusion_matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Real", "Predicted Fake"])
    ax.set_yticklabels(["Actual Real", "Actual Fake"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{confusion_matrix[row, col]:,}",
                ha="center",
                va="center",
                fontsize=12
            )

    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Matrix layout: rows = actual class, columns = predicted class. "
        "Evaluation set: 24,844 samples."
    )

    st.divider()
    st.subheader("📋 Detailed Metrics")

    st.dataframe(
        {
            "Metric": [
                "Accuracy",
                "Real Precision",
                "Real Recall",
                "Real F1 Score",
                "Fake Precision",
                "Fake Recall",
                "Fake F1 Score"
            ],
            "Score": [
                f"{accuracy:.2f}%",
                f"{real_precision:.2f}%",
                f"{real_recall:.2f}%",
                f"{real_f1:.2f}%",
                f"{fake_precision:.2f}%",
                f"{fake_recall:.2f}%",
                f"{fake_f1:.2f}%"
            ]
        },
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "Important: Model predictions are not a guarantee that audio is "
        "genuine or manipulated. Results can vary with audio quality, "
        "recording conditions and unseen voice-generation methods."
    )

import streamlit as st
import pandas as pd

from src.history import get_user_history


def render_analysis_history(username):
    st.markdown("## 📜 ANALYSIS HISTORY")
    st.caption("Review your previously analyzed audio files.")

    if not username:
        st.warning("Please log in to view your analysis history.")
        return

    try:
        records = get_user_history(username, limit=500)
    except Exception as error:
        st.error(f"Could not load analysis history: {error}")
        return

    if not records:
        st.info(
            "No analysis history found yet. "
            "Analyze an audio file while logged in, and it will appear here."
        )
        return

    df = pd.DataFrame(records)

    total = len(df)
    real_count = int((df["prediction"] == "Real").sum())
    fake_count = int((df["prediction"] == "Fake").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("TOTAL ANALYSES", total)
    col2.metric("REAL DETECTIONS", real_count)
    col3.metric("FAKE DETECTIONS", fake_count)

    st.divider()

    search = st.text_input(
        "🔎 Search by audio filename",
        key="history_filename_search",
        placeholder="Enter filename..."
    )

    if search.strip():
        df = df[
            df["filename"].str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    if df.empty:
        st.info("No records match your search.")
        return

    display_df = df[
        [
            "created_at",
            "filename",
            "prediction",
            "confidence",
            "real_probability",
            "fake_probability",
            "duration",
        ]
    ].copy()

    display_df.columns = [
        "Date & Time",
        "Audio Filename",
        "Prediction",
        "Confidence",
        "Real Probability",
        "Fake Probability",
        "Duration (seconds)",
    ]

    for column in [
        "Confidence",
        "Real Probability",
        "Fake Probability",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: f"{float(value):.2%}"
        )

    display_df["Duration (seconds)"] = display_df[
        "Duration (seconds)"
    ].map(
        lambda value: (
            f"{float(value):.2f}" if pd.notna(value) else "N/A"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download History as CSV",
        data=csv_data,
        file_name="sentinel_ai_analysis_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Note: Predictions are model estimates, not proof that an audio "
        "recording is authentic or manipulated."
    )

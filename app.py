import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path

from src.prediction import predict_audio
from src.pdf_report import generate_pdf_report
from src.auth import register_user, verify_user
from src.history import save_analysis, get_user_history
from src.history_ui import render_analysis_history
from src.performance_dashboard import render_performance_dashboard


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentinel AI | Deepfake Audio Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CYBERSECURITY UI
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #050807;
    color: #e5ffe9;
}

[data-testid="stSidebar"] {
    background: #09120e;
    border-right: 1px solid #173b25;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 32px;
    border-radius: 20px;
    background: linear-gradient(125deg, #0a1910, #10251a, #07120c);
    border: 1px solid #235b36;
    margin-bottom: 25px;
    box-shadow: 0 0 28px rgba(42,255,113,0.07);
}

.hero h1 {
    color: #55ff91;
    font-size: 36px;
    font-weight: 800;
}

.hero p {
    color: #a9cbb4;
    font-size: 15px;
}

.panel {
    background: #0a120e;
    border: 1px solid #1e432b;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}

.result-real {
    background: linear-gradient(135deg, #071e12, #0b2b19);
    border: 1px solid #28e878;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.result-fake {
    background: linear-gradient(135deg, #260b0b, #350c13);
    border: 1px solid #ff4d62;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.result-title {
    font-size: 12px;
    letter-spacing: 3px;
    color: #b8d9c1;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    margin: 12px 0;
    color: white;
}

.result-subtitle {
    color: #c4d8ca;
    font-size: 14px;
}

div[data-testid="stMetric"] {
    background: #0a120e;
    border: 1px solid #1e432b;
    padding: 16px;
    border-radius: 13px;
}

div[data-testid="stMetricLabel"] {
    color: #9fbea9;
}

div[data-testid="stMetricValue"] {
    color: #55ff91;
}

.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 10px;
    font-weight: 700;
    border: 1px solid #36d96d;
    background: #10291a;
    color: #e5ffe9;
}

.stButton > button:hover {
    background: #174a2a;
    border-color: #55ff91;
    color: white;
}

h1, h2, h3 {
    color: #e5ffe9;
}

hr {
    border-color: #1e432b;
}

[data-testid="stFileUploader"] {
    background: #0a120e;
    border: 1px dashed #28643b;
    border-radius: 14px;
    padding: 12px;
}

.small-note {
    color: #8ba795;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "show_auth" not in st.session_state:
    st.session_state.show_auth = False


# ============================================================
# AUTHENTICATION DIALOG
# ============================================================

@st.dialog("🔐 Sign in to Sentinel AI")
def authentication_dialog():

    st.caption(
        "Create an account or sign in to analyze audio."
    )

    login_tab, signup_tab = st.tabs([
        "LOGIN",
        "CREATE ACCOUNT",
    ])

    with login_tab:
        with st.form("public_login_form"):
            login_username = st.text_input(
                "Username",
                key="public_login_username",
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="public_login_password",
            )

            login_submit = st.form_submit_button(
                "LOGIN",
                use_container_width=True,
            )

        if login_submit:
            if not login_username.strip() or not login_password:
                st.error("Enter your username and password.")

            elif verify_user(login_username, login_password):
                st.session_state.authenticated = True
                st.session_state.username = login_username.strip()
                st.session_state.show_auth = False
                st.rerun()

            else:
                st.error("Invalid username or password.")

    with signup_tab:
        with st.form("public_signup_form"):
            signup_username = st.text_input(
                "Choose a username",
                key="public_signup_username",
            )

            signup_password = st.text_input(
                "Create password",
                type="password",
                key="public_signup_password",
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="public_confirm_password",
            )

            signup_submit = st.form_submit_button(
                "CREATE ACCOUNT",
                use_container_width=True,
            )

        if signup_submit:
            if signup_password != confirm_password:
                st.error("Passwords do not match.")

            else:
                success, message = register_user(
                    signup_username,
                    signup_password,
                )

                if success:
                    st.success(message)
                    st.info("Switch to LOGIN to access audio analysis.")
                else:
                    st.error(message)


# ============================================================
# AUTHENTICATION HELPER
# ============================================================

def require_login():
    if st.session_state.authenticated:
        return True

    st.warning(
        "🔒 Login required. Create an account or sign in "
        "to analyze this audio."
    )

    if st.button(
        "🔐 LOGIN / SIGN UP TO ANALYZE",
        key="open_login_from_analysis",
        use_container_width=True,
    ):
        st.session_state.show_auth = True
        st.rerun()

    return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ SENTINEL AI")
    st.caption("DEEPFAKE AUDIO INTELLIGENCE")

    st.divider()

    if st.session_state.authenticated:
        st.success(
            f"Logged in as: {st.session_state.username}"
        )

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.show_auth = False
            st.rerun()

    else:
        st.info("Guest access")

        if st.button(
            "🔐 Login / Sign Up",
            use_container_width=True,
        ):
            st.session_state.show_auth = True
            st.rerun()

    st.divider()

    st.markdown("### MODEL INFORMATION")
    st.write("**Architecture:** CNN")
    st.write("**Dataset:** ASVspoof 2019 LA")
    st.write("**Input:** Upload + Microphone")
    st.write("**Fake threshold:** 0.90")

    st.divider()

    st.markdown("### SUPPORTED FORMATS")
    st.write("WAV · FLAC · MP3 · OGG")

    st.info(
        "Predictions are estimates and cannot guarantee "
        "that audio is authentic."
    )


# ============================================================
# PAGE NAVIGATION
# ============================================================

with st.sidebar:
    st.divider()
    st.markdown("### NAVIGATION")

    selected_page = st.radio(
        "Choose section",
        ["?? Dashboard", "?? Analysis History", "?? Model Performance"],
        key="sentinel_page_navigation",
        label_visibility="collapsed",
    )


# MODEL PERFORMANCE PAGE
if selected_page == "?? Model Performance":
    render_performance_dashboard()
    st.stop()

# ANALYSIS HISTORY PAGE
# ============================================================

if selected_page == "?? Analysis History":
    st.markdown("# ?? ANALYSIS HISTORY")
    st.caption("Your previous audio analysis results.")

    if st.session_state.get("authenticated") and st.session_state.get("username"):
        render_analysis_history(st.session_state.username)
    else:
        st.warning("?? Please log in to view your analysis history.")
        if st.button("?? LOGIN / SIGN UP", key="history_login_button"):
            st.session_state.show_auth = True

    if st.session_state.get("show_auth") and not st.session_state.get("authenticated"):
        authentication_dialog()

    st.stop()


# MAIN DASHBOARD
# ============================================================

st.markdown("""
<div class="hero">
    <p style="letter-spacing:3px;color:#55ff91;font-weight:700;">
        AI SECURITY / AUDIO FORENSICS
    </p>
    <h1>🛡️ SENTINEL AI</h1>
    <p>
        Deepfake Audio Detection System<br>
        Analyze speech to identify potentially spoofed audio.
    </p>
    <p>
        Public dashboard · Secure analysis access
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("AI ENGINE", "CNN", "Trained model")

with c2:
    st.metric("INPUT MODES", "02", "Upload + Microphone")

with c3:
    if st.session_state.authenticated:
        st.metric("ACCESS STATUS", "SIGNED IN", "Analysis enabled")
    else:
        st.metric("ACCESS STATUS", "GUEST", "Login to analyze")


# ============================================================
# AUDIO ANALYSIS FUNCTION
# ============================================================

def analyze_audio_file(audio_path, source_name):

    fig = None

    try:
        with st.spinner("Analyzing audio with the trained CNN..."):

            result = predict_audio(audio_path)

            # Save successful analysis for the authenticated user
            if st.session_state.get("authenticated") and st.session_state.get("username"):
                try:
                    save_analysis(
                        username=st.session_state.username,
                        filename=Path(str(source_name)).name,
                        prediction=result["prediction"],
                        confidence=result["confidence"],
                        real_probability=result["real_probability"],
                        fake_probability=result["fake_probability"],
                        duration=librosa.get_duration(path=str(audio_path)),
                    )
                except Exception as history_error:
                    print(f"History save failed: {history_error}")

            audio, sr = librosa.load(
                str(audio_path),
                sr=16000,
                mono=True,
                duration=5,
            )

            duration = librosa.get_duration(
                path=str(audio_path)
            )

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=1024,
                hop_length=256,
                n_mels=64,
                power=2.0,
            )

            log_mel = librosa.power_to_db(
                mel,
                ref=np.max,
            )

        st.divider()
        st.markdown("## 🧠 ANALYSIS RESULTS")

        if result["prediction"] == "Real":
            st.markdown("""
            <div class="result-real">
                <div class="result-title">CLASSIFICATION RESULT</div>
                <div class="result-value">✓ REAL AUDIO</div>
                <div class="result-subtitle">
                    The model classified this audio as Real.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-fake">
                <div class="result-title">CLASSIFICATION RESULT</div>
                <div class="result-value">⚠ FAKE AUDIO</div>
                <div class="result-subtitle">
                    The model classified this audio as Fake.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Model Confidence",
            f'{result["confidence"]:.2%}',
        )

        col2.metric(
            "Real Probability",
            f'{result["real_probability"]:.2%}',
        )

        col3.metric(
            "Fake Probability",
            f'{result["fake_probability"]:.2%}',
        )

        st.markdown("### 📊 CLASS PROBABILITIES")

        st.write("Real audio probability")

        st.progress(
            max(
                0.0,
                min(1.0, float(result["real_probability"])),
            )
        )

        st.write("Fake audio probability")

        st.progress(
            max(
                0.0,
                min(1.0, float(result["fake_probability"])),
            )
        )

        st.caption(
            f'Fake classification threshold: '
            f'{result.get("threshold", 0.90):.2f}'
        )

        st.markdown("## 🌊 AUDIO SPECTROGRAM")

        fig, ax = plt.subplots(figsize=(12, 4))

        librosa.display.specshow(
            log_mel,
            sr=sr,
            hop_length=256,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )

        ax.set_title("Log-Mel Spectrogram")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (Mel)")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)
        fig = None

        st.markdown("## 📋 ANALYSIS SUMMARY")

        st.write(f"**Source:** {source_name}")
        st.write(f"**Duration:** {duration:.2f} seconds")
        st.write(f'**Prediction:** {result["prediction"]}')
        st.write(f'**Confidence:** {result["confidence"]:.2%}')

        # Generate a downloadable PDF report for this analysis
        try:
            pdf_bytes = generate_pdf_report(
                filename=Path(str(source_name)).name,
                prediction=result["prediction"],
                confidence=result["confidence"],
                real_probability=result["real_probability"],
                fake_probability=result["fake_probability"],
                duration=duration,
                sample_rate=sr,
            )

            st.download_button(
                label="?? DOWNLOAD PDF REPORT",
                data=pdf_bytes,
                file_name="sentinel_ai_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_pdf_report",
            )
        except Exception as pdf_error:
            st.error(f"Could not generate PDF report: {pdf_error}")

        st.warning(
            "This is an AI prediction, not proof of authenticity. "
            "Results may differ on unfamiliar real-world audio."
        )

    except Exception as error:
        st.error(f"Could not analyze audio: {error}")

    finally:
        if fig is not None:
            plt.close(fig)


# ============================================================
# INPUT TABS
# ============================================================

upload_tab, microphone_tab = st.tabs([
    "📁 UPLOAD AUDIO",
    "🎙️ MICROPHONE ANALYSIS",
])


# ============================================================
# UPLOAD AUDIO
# ============================================================

with upload_tab:

    st.markdown("### Analyze an audio file")
    st.caption("WAV, FLAC, MP3 and OGG are supported.")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac", "mp3", "ogg"],
        key="audio_upload",
    )

    if uploaded_file is not None:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.write("**Selected file:**", uploaded_file.name)

        st.write(
            "**File size:**",
            f"{uploaded_file.size / 1024:.2f} KB",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.audio(uploaded_file)

        if st.button(
            "🔍 ANALYZE AUDIO",
            type="primary",
            key="analyze_upload",
            use_container_width=True,
        ):

            # Authentication is checked before model inference.
            if require_login():

                temp_path = None

                try:
                    suffix = Path(uploaded_file.name).suffix or ".wav"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = Path(temp_file.name)

                    analyze_audio_file(
                        temp_path,
                        uploaded_file.name,
                    )

                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)

    else:

        st.markdown("""
        <div class="panel">
            <h3>◈ Awaiting audio input</h3>
            <p>Upload a file to preview it before analysis.</p>
            <p class="small-note">
                You can browse the dashboard without an account.
                Login is required only for analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MICROPHONE RECORDING
# ============================================================

with microphone_tab:

    st.markdown("### Record speech for analysis")

    st.caption(
        "Allow microphone access and record a speech sample."
    )

    recorded_audio = st.audio_input(
        "Record your voice",
        sample_rate=16000,
        key="microphone_recording",
    )

    if recorded_audio is not None:

        st.audio(recorded_audio)

        if st.button(
            "🎙️ ANALYZE RECORDING",
            type="primary",
            key="analyze_microphone",
            use_container_width=True,
        ):

            # Authentication is checked before model inference.
            if require_login():

                microphone_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav",
                    ) as temp_file:

                        temp_file.write(recorded_audio.getvalue())
                        microphone_path = Path(temp_file.name)

                    audio_info = sf.info(str(microphone_path))

                    st.markdown("### MICROPHONE DIAGNOSTICS")

                    d1, d2, d3, d4 = st.columns(4)

                    d1.metric(
                        "Sample Rate",
                        f"{audio_info.samplerate} Hz",
                    )

                    d2.metric(
                        "Channels",
                        str(audio_info.channels),
                    )

                    d3.metric(
                        "Duration",
                        f"{audio_info.duration:.2f} sec",
                    )

                    d4.metric(
                        "Format",
                        str(audio_info.format),
                    )

                    samples, sample_rate = sf.read(
                        str(microphone_path),
                        dtype="float32",
                    )

                    if samples.ndim > 1:
                        mono_samples = samples.mean(axis=1)
                    else:
                        mono_samples = samples

                    audio_rms = (
                        float(np.sqrt(np.mean(np.square(mono_samples))))
                        if len(mono_samples)
                        else 0.0
                    )

                    audio_peak = (
                        float(np.max(np.abs(mono_samples)))
                        if len(mono_samples)
                        else 0.0
                    )

                    st.write(f"**Audio RMS level:** {audio_rms:.5f}")
                    st.write(f"**Audio peak level:** {audio_peak:.5f}")

                    if audio_info.duration < 1:
                        st.warning(
                            "Recording is shorter than one second. "
                            "Record a longer speech sample."
                        )

                    if audio_rms < 0.005:
                        st.warning(
                            "Audio level is very low. Check your "
                            "microphone and record again if necessary."
                        )

                    analyze_audio_file(
                        microphone_path,
                        "Microphone recording",
                    )

                except Exception as error:
                    st.error(
                        f"Could not process microphone audio: {error}"
                    )

                finally:
                    if microphone_path is not None:
                        microphone_path.unlink(missing_ok=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:12px;color:#6f907a;">
    <p style="font-size:13px;">
        🛡️ SENTINEL AI · DEEPFAKE AUDIO DETECTION
    </p>
    <p style="font-size:11px;">
        AI predictions can be incorrect. Do not use this result as the
        sole basis for security, legal, or identity decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOGIN DIALOG
# ============================================================

if st.session_state.show_auth and not st.session_state.authenticated:
    authentication_dialog()



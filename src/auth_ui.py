import streamlit as st
from src.auth import register_user, verify_user


def render_auth_screen():
    st.markdown("""
    <style>
    .stApp {
        background: #050807;
        color: #e5ffe9;
    }
    [data-testid="stSidebar"] {
        background: #09120e;
    }
    .auth-hero {
        max-width: 720px;
        margin: 35px auto 20px auto;
        padding: 30px;
        text-align: center;
        background: linear-gradient(135deg, #0b1b12, #10251a);
        border: 1px solid #1d653a;
        border-radius: 20px;
        box-shadow: 0 0 28px rgba(0, 255, 100, 0.08);
    }
    .auth-hero h1 {
        color: #55ff91;
        font-size: 34px;
        font-weight: 800;
    }
    .auth-hero p {
        color: #a7cbb2;
        font-size: 15px;
    }
    div[data-testid="stForm"] {
        background: #0a120e;
        border: 1px solid #20462e;
        border-radius: 16px;
        padding: 24px;
    }
    .stButton > button {
        border: 1px solid #36d96d;
        border-radius: 9px;
        min-height: 44px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-hero">
        <h1>⌁ SENTINEL AI</h1>
        <p>DEEPFAKE AUDIO INTELLIGENCE SYSTEM</p>
        <p>Secure access · AI-powered audio analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("authenticated", False):
        return True

    login_tab, signup_tab = st.tabs(["🔐 LOGIN", "＋ CREATE ACCOUNT"])

    with login_tab:
        st.markdown("### Welcome back")
        st.caption("Sign in to access the audio analysis dashboard.")

        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )
            login_clicked = st.form_submit_button(
                "ACCESS DASHBOARD",
                use_container_width=True,
            )

        if login_clicked:
            if not username.strip() or not password:
                st.error("Enter both username and password.")
            elif verify_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with signup_tab:
        st.markdown("### Create your account")
        st.caption("Register to use the deepfake audio detector.")

        with st.form("signup_form"):
            new_username = st.text_input(
                "Choose a username",
                key="signup_username",
            )
            new_password = st.text_input(
                "Create password",
                type="password",
                key="signup_password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="signup_confirm_password",
            )
            signup_clicked = st.form_submit_button(
                "CREATE ACCOUNT",
                use_container_width=True,
            )

        if signup_clicked:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(
                    new_username,
                    new_password,
                )
                if success:
                    st.success(message)
                    st.info("Open the LOGIN tab to sign in.")
                else:
                    st.error(message)

    st.markdown(
        "<p style='text-align:center;color:#64816d;font-size:12px;'>"
        "AUTHORIZED ACCESS ONLY · SENTINEL AI</p>",
        unsafe_allow_html=True,
    )
    return False

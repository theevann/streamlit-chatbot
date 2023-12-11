import hmac
import streamlit as st


def check_password():
    """Returns `True` if the user had the correct password."""

    if st.session_state.get("password_correct", False):
        return True

    def password_entered():
        st.session_state["password_correct"] = hmac.compare_digest(st.session_state["password"], st.secrets["PASSWORD"])

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
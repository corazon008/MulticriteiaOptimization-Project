import streamlit as st


def init_state():
    st.session_state.setdefault("model", "Markowitz")

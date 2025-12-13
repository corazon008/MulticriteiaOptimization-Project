import streamlit as st
from streamlit_tools.app_utils import calculate_portfolio


@st.cache_data
def compute_level2_frontier(mu, sigma, w0, K, c):
    return calculate_portfolio(mu, sigma, K=K, c=c, w0=w0)

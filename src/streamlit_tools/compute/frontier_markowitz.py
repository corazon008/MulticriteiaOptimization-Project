import streamlit as st
from streamlit_tools.app_utils import calculate_markowitz_frontier


@st.cache_data
def compute_markowitz_frontier(mu, sigma):
    return calculate_markowitz_frontier(mu, sigma)

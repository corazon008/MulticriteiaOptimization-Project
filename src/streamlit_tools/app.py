import numpy as np
import streamlit as st
from data.market_data import load_market_data
from compute.frontier_markowitz import compute_markowitz_frontier
from compute.frontier_level2 import compute_level2_frontier
from plots.frontier_2d import plot_frontier_2d
from plots.frontier_3d import plot_frontier_3d

from ui.sidebar import render_sidebar
from ui.details import render_details
from state import init_state
from graph import plot_level2_frontier

st.set_page_config(layout="wide")
init_state()

df_prices, mu, sigma, sector_map = load_market_data()
w0 = np.zeros_like(len(mu))

model, r_min, K, c = render_sidebar(mu)

if model == "Markowitz":
    df_frontier = compute_markowitz_frontier(mu, sigma)
else:
    df_frontier = compute_level2_frontier(mu, sigma,w0, K, c)

valid = df_frontier[df_frontier["return"] >= r_min]
best = valid.loc[valid["volatility"].idxmin()] if not valid.empty else None

has_solution = best is not None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Frontière Efficiente")
    if model == "Markowitz":
        fig = plot_frontier_2d(df_frontier, r_min, has_solution, best)
        st.plotly_chart(fig, use_container_width=True)

    else:
        fig = plot_frontier_3d(df_frontier, has_solution, best)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    render_details(best, mu, sector_map, K or 5)

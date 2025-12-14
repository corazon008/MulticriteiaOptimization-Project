import numpy as np
import streamlit as st
from data.market_data import load_market_data
from compute.frontier_markowitz import compute_markowitz_frontier
from compute.frontier_level2 import compute_level2_frontier
from level3.functions import PortfolioRobustness
from plots.frontier_2d import plot_frontier_2d
from plots.frontier_3d import plot_frontier_3d

from ui.sidebar import render_sidebar
from ui.details import render_details
from state import init_state
from pages.home import render_home
from pages.settings import render_settings

st.set_page_config(layout="wide")
init_state()

# Chargement des données de marché
df_prices, mu, sigma, sector_map = load_market_data()
asset_names = list(df_prices.columns)

# Navigation entre pages
page = st.sidebar.radio("Navigation", options=["Accueil", "Paramètres", "Modèles"], index=["Accueil", "Paramètres", "Modèles"].index(st.session_state.get("page", "Accueil")))
st.session_state["page"] = page

if page == "Accueil":
    render_home(asset_names)
    st.stop()

if page == "Paramètres":
    render_settings()
    st.stop()

# Si on arrive ici, on est sur la page "Modèles"
# Récupérer W0 depuis la session si défini, sinon portefeuille vide par défaut
w0 = st.session_state.get("w0_array")
if w0 is None:
    num_assets = df_prices.shape[1]
    w0 = np.zeros(num_assets)

# Affichage des options du modèle
model, r_min, K, c = render_sidebar(mu)

# Défaut K si non défini
K_eff = K or 5

# Créer l'objet de robustesse du portefeuille
portfolio_robustness = PortfolioRobustness(df_prices, w0, K_eff, 0, c=c if c is not None else 0.01)

# Calcul des frontières selon le modèle
if model == "Markowitz":
    df_frontier = compute_markowitz_frontier(mu, sigma)
else:
    df_frontier = compute_level2_frontier(mu, sigma, w0, K, c)

valid = df_frontier[df_frontier["return"] >= r_min]
best = valid.loc[valid["volatility"].idxmin()] if not valid.empty else None

has_solution = best is not None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Frontière Efficiente")

    # Si on a des poids dans valid, on peut faire un skip pour la robustesse
    try:
        portfolio_robustness.skip_optimize(valid["weights"].to_numpy())
    except Exception:
        pass

    if model == "Markowitz":
        fig = plot_frontier_2d(df_frontier, r_min, has_solution, best)
    else:
        fig = plot_frontier_3d(df_frontier, has_solution, best)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    render_details(best, mu, sector_map, K_eff, portfolio_robustness)

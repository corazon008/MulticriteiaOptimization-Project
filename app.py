import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Imports de vos modules
from src.portfolio_utils import load_datas, f_mu_on_df, f_sigma_on_df
from src.app_utils import get_ticker_sector_map, calculate_markowitz_frontier, load_saved_frontier

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Optimisation de Portefeuille",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- CHARGEMENT DES DONNÉES (Mise en cache pour la rapidité) ---
@st.cache_data
def get_market_data():
    """Charge les données de marché brutes et calcule Mu/Sigma."""
    df_prices = load_datas()
    returns = np.log(df_prices / df_prices.shift(1)).dropna()
    mu = f_mu_on_df(returns)
    sigma = f_sigma_on_df(returns)
    sector_map = get_ticker_sector_map("datasets")
    return df_prices, mu, sigma, sector_map


# Chargement initial
try:
    with st.spinner('Chargement des données de marché...'):
        df_prices, mu, sigma, sector_map = get_market_data()
except Exception as e:
    st.error(f"Erreur critique lors du chargement des données : {e}")
    st.stop()

# --- SIDEBAR : CONTRÔLES ---
st.sidebar.header("🎯 Paramètres de Décision")

# 1. Choix du Modèle
st.sidebar.subheader("1. Modèle d'Optimisation")
model_choice = st.sidebar.radio(
    "Source des données :",
    ("Markowitz (Niveau 1)", "Contraintes & Coûts (Niveau 2)")
)

# 2. Contrainte Utilisateur
st.sidebar.subheader("2. Contraintes")
min_ret_possible = max(0.00, float(mu.min()))
max_ret_possible = float(mu.max())
default_val = float((min_ret_possible + max_ret_possible) / 2)

r_min = st.sidebar.slider(
    "Rendement Minimal Souhaité ($r_{min}$)",
    min_value=min_ret_possible,
    max_value=max_ret_possible,
    value=default_val,
    format="%.4f"
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Guide Rapide :**
    1. Choisissez un modèle.
    2. Ajustez le rendement min.
    3. Le point **ROUGE** sur le graphique indique le portefeuille optimal (risque minimal) pour ce rendement.
    """
)

# --- LOGIQUE DE RÉCUPÉRATION DES FRONTIÈRES ---
# On prépare le DataFrame 'df_frontier' selon le choix utilisateur

if "Markowitz" in model_choice:
    # On calcule ou on charge si existe (optionnel, ici on calcule pour être sûr d'avoir une belle courbe)
    if 'df_markowitz' not in st.session_state:
        st.session_state['df_markowitz'] = calculate_markowitz_frontier(mu, sigma)
    df_frontier = st.session_state['df_markowitz']
    source_name = "Markowitz (Niveau 1)"

else:
    # Chargement niveau 2
    pickle_path = "notebooks/frontier_data_2.pkl"  # Vérifiez ce nom de fichier !
    df_loaded = load_saved_frontier(pickle_path)

    if df_loaded is not None and not df_loaded.empty:
        df_frontier = df_loaded
        source_name = "Contraintes (Niveau 2)"
    else:
        st.warning(f"⚠️ Fichier `{pickle_path}` introuvable ou vide. Affichage de Markowitz par défaut.")
        if 'df_markowitz' not in st.session_state:
            st.session_state['df_markowitz'] = calculate_markowitz_frontier(mu, sigma)
        df_frontier = st.session_state['df_markowitz']
        source_name = "Markowitz (Fallback)"

# --- MAIN : ANALYSE ET SÉLECTION ---

st.title("📊 Tableau de Bord d'Allocation")

# Layout Principal : Gauche (Graphique Frontière) / Droite (Détails Portefeuille)
col_left, col_right = st.columns([2, 1])

# --- LOGIQUE DE SÉLECTION DU PORTEFEUILLE OPTIMAL ---
# Filtre : Rendement >= r_min
valid_portfolios = df_frontier[df_frontier['return'] >= r_min]

if not valid_portfolios.empty:
    # Sélection : Celui qui minimise la volatilité parmi les valides
    best_idx = valid_portfolios['volatility'].idxmin()
    best_portfolio = valid_portfolios.loc[best_idx]
    has_solution = True
else:
    best_portfolio = None
    has_solution = False

# --- COLONNE GAUCHE : FRONTIÈRE DE PARETO ---
with col_left:
    st.subheader(f"Frontière Efficiente : {source_name}")

    # Construction du graphique
    fig = go.Figure()

    # 1. Tous les points (Frontière)
    fig.add_trace(go.Scatter(
        x=df_frontier['volatility'],
        y=df_frontier['return'],
        mode='markers',
        name='Portefeuilles testés',
        marker=dict(color='royalblue', size=8, opacity=0.6),
        hovertemplate='<b>Vol:</b> %{x:.2%}<br><b>Rend:</b> %{y:.2%}<extra></extra>'
    ))

    # 2. Ligne de seuil r_min
    fig.add_hline(y=r_min, line_dash="dash", line_color="gray", annotation_text=f"Min: {r_min:.2%}")

    # 3. Point Optimal (si solution trouvée)
    if has_solution:
        fig.add_trace(go.Scatter(
            x=[best_portfolio['volatility']],
            y=[best_portfolio['return']],
            mode='markers',
            name='Portefeuille Choisi',
            marker=dict(color='red', size=15, symbol='star'),
            hovertemplate='<b>CHOIX OPTIMAL</b><br>Vol: %{x:.2%}<br>Rend: %{y:.2%}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Risque (Volatilité)",
        yaxis_title="Rendement Espéré",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# --- COLONNE DROITE : DÉTAILS DU SÉLECTIONNÉ ---
with col_right:
    st.subheader("🔎 Détails de la Sélection")

    if has_solution:
        # 1. KPIs
        c1, c2 = st.columns(2)
        c1.metric("Rendement", f"{best_portfolio['return']:.2%}", delta_color="normal")
        c2.metric("Volatilité", f"{best_portfolio['volatility']:.2%}", delta_color="inverse")

        sharpe = best_portfolio['return'] / best_portfolio['volatility'] if best_portfolio['volatility'] > 0 else 0
        st.metric("Ratio de Sharpe", f"{sharpe:.2f}")

        # 2. Analyse Structurelle (Macro)
        st.markdown("#### 🏗️ Allocation Macro-économique")

        # Récupération et nettoyage des poids
        weights = best_portfolio['weights']
        asset_names = mu.index.tolist()

        # Gestion de formats (si weights est array ou liste)
        if len(weights) == len(asset_names):
            df_w = pd.DataFrame({'Ticker': asset_names, 'Poids': weights})
        else:
            # Cas dégradé (taille différente)
            st.warning("Dimension des poids incohérente avec les données.")
            df_w = pd.DataFrame({'Ticker': [f'A{i}' for i in range(len(weights))], 'Poids': weights})

        # Ajout du secteur via le mapping
        df_w['Secteur'] = df_w['Ticker'].map(sector_map).fillna('Indeterminé')

        # On filtre les poids négligeables pour la clarté
        df_active = df_w[df_w['Poids'] > 0.001].copy()  # > 0.1%

        # Agrégation par Secteur
        df_sector = df_active.groupby('Secteur')['Poids'].sum().reset_index()

        # Graphique Camembert Sectoriel
        fig_pie = px.pie(
            df_sector,
            values='Poids',
            names='Secteur',
            title="Exposition Sectorielle",
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Petit tableau des top positions
        st.markdown("#### 🏆 Top 5 Actifs")
        top_assets = df_active.sort_values(by='Poids', ascending=False).head(5)
        st.dataframe(
            top_assets[['Ticker', 'Secteur', 'Poids']].style.format({'Poids': '{:.2%}'}),
            hide_index=True
        )

    else:
        st.error(f"❌ Impossible ! Aucun portefeuille n'atteint {r_min:.2%} de rendement.")
        st.markdown("👉 Veuillez diminuer le rendement minimal demandé dans la barre latérale.")
import streamlit as st
import pandas as pd
import plotly.express as px

from level3.functions import PortfolioRobustness


def render_details(best, mu, sector_map, K, portfolio_robustness:PortfolioRobustness):
    if best is None:
        st.error("Aucune solution")
        return

    c1, c2 = st.columns(2)
    c1.metric("Rendement", f"{best['return']:.2%}")
    c2.metric("Volatilité", f"{best['volatility']:.2%}")

    sharpe_c, score_c = st.columns(2)
    sharpe = best['return'] / best['volatility'] if best['volatility'] > 0 else 0
    sharpe_c.metric("Ratio de Sharpe", f"{sharpe:.2f}")

    score = portfolio_robustness.compute_score(best['weights'], 0.7, 0.3)
    score_c.metric("Robustesse", f"{score:.2f}")

    # 2. Analyse Structurelle (Macro)
    st.markdown("#### 🏗️ Allocation Macro-économique")

    # Récupération et nettoyage des poids
    weights = best['weights']
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
    st.markdown(f"#### 🏆 Top {K} Actifs")
    top_assets = df_active.sort_values(by='Poids', ascending=False).head(K)
    st.dataframe(
        top_assets[['Ticker', 'Secteur', 'Poids']].style.format({'Poids': '{:.2%}'}),
        hide_index=True
    )

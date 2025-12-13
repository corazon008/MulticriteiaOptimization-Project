import plotly.graph_objects as go

def plot_markowitz_frontier(df_frontier, r_min, has_solution, best_portfolio):
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
    return fig

def plot_level2_frontier(df_frontier, has_solution, best_portfolio):
    fig = go.Figure()

    # 1. Tous les points (Frontière)
    fig.add_trace(go.Scatter3d(
        x=df_frontier['volatility'],
        y=df_frontier['return'],
        z=df_frontier['cost'],
        mode='markers',
        name='Portefeuilles testés',
        marker=dict(color='royalblue', size=8, opacity=0.6),
        hovertemplate='<b>Vol:</b> %{x:.2%}<br><b>Rend:</b> %{y:.2%}<extra></extra>'
    ))

    # 3. Point Optimal (si solution trouvée)
    if has_solution:
        fig.add_trace(go.Scatter3d(
            x=[best_portfolio['volatility']],
            y=[best_portfolio['return']],
            z=[best_portfolio['cost']],
            mode='markers',
            name='Portefeuille Choisi',
            marker=dict(color='red', size=15),
            hovertemplate='<b>CHOIX OPTIMAL</b><br>Vol: %{x:.2%}<br>Rend: %{y:.2%}<extra></extra>'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Risque (Volatilité)",
            yaxis_title="Rendement Espéré",
            zaxis_title="Coût de Transaction",
            camera=dict(
                eye=dict(x=-1.8, y=-0.8, z=0.6)  # 👈 vue frontale basse
            )
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01
        ),
        height=1000
    )
    return fig

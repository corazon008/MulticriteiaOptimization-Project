import plotly.graph_objects as go


def plot_frontier_2d(df_frontier, r_min, has_solution, best):
    fig = go.Figure()

    # Frontière
    fig.add_trace(go.Scatter(
        x=df_frontier["volatility"],
        y=df_frontier["return"],
        mode="markers",
        name="Frontière",
        marker=dict(size=7, opacity=0.7)
    ))

    # Ligne r_min
    fig.add_hline(
        y=r_min,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"r_min = {r_min:.2%}"
    )

    # Point optimal
    if has_solution:
        fig.add_trace(go.Scatter(
            x=[best["volatility"]],
            y=[best["return"]],
            mode="markers",
            marker=dict(size=14, color="red"),
            name="Optimal"
        ))

    fig.update_layout(
        xaxis_title="Risque (Volatilité)",
        yaxis_title="Rendement Espéré",
        height=500
    )

    return fig

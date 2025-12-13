import plotly.graph_objects as go


def plot_frontier_3d(df_frontier, has_solution, best):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df_frontier["volatility"],
        y=df_frontier["return"],
        z=df_frontier["cost"],
        mode="markers",
        marker=dict(size=6, opacity=0.6),
        name="Frontière"
    ))

    if has_solution:
        fig.add_trace(go.Scatter3d(
            x=[best["volatility"]],
            y=[best["return"]],
            z=[best["cost"]],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Optimal"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Risque",
            yaxis_title="Rendement",
            zaxis_title="Coût",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.5))
        ),
        height=1000
    )
    return fig
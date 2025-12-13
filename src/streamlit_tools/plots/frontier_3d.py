import plotly.graph_objects as go


def plot_frontier_3d(df_frontier, has_solution:bool, best)-> go.Figure:
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
            #camera=dict(eye=dict(x=-1.8, y=-0.8, z=0.6))
        ),
        height=1000
    )
    return fig
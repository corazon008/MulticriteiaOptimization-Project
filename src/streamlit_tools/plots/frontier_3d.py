import plotly.graph_objects as go
import pandas as pd


def plot_frontier_3d(df_frontier, has_solution, best):
    fig = go.Figure()

    # Sécurité : s'assurer qu'on a bien un DataFrame
    if df_frontier is None:
        fig.add_annotation(text="Aucune donnée pour la frontière (df_frontier is None)", showarrow=False)
        return fig

    # Copie pour ne pas muter l'original
    df = df_frontier.copy()

    # Forcer conversion des colonnes attendues en numériques
    for col in ("volatility", "return", "cost"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Colonne manquante → message et sortie
            fig.add_annotation(text=f"Colonne manquante: {col}", showarrow=False)
            return fig

    # Supprimer les lignes invalides
    df = df.dropna(subset=["volatility", "return", "cost"]).reset_index(drop=True)

    if df.empty:
        fig.add_annotation(text="Aucune solution valide pour tracer la frontière (lignes supprimées ou DataFrame vide)", showarrow=False)
        # Si on a une solution optimale, on peut tout de même la tracer si elle est complète
        if has_solution and best is not None:
            try:
                fig.add_trace(go.Scatter3d(
                    x=[float(best["volatility"])],
                    y=[float(best["return"])],
                    z=[float(best["cost"])],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name="Optimal"
                ))
            except Exception:
                pass
        return fig

    # Convertir en listes simples pour plotly (évite les objets numpy dans les cellules)
    x = df["volatility"].astype(float).tolist()
    y = df["return"].astype(float).tolist()
    z = df["cost"].astype(float).tolist()

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=6, opacity=0.8, color='blue'),
        name="Frontière"
    ))

    # Point optimal
    if has_solution and best is not None:
        try:
            bx = float(best["volatility"]) if "volatility" in best else None
            by = float(best["return"]) if "return" in best else None
            bz = float(best["cost"]) if "cost" in best else None
            if None not in (bx, by, bz):
                fig.add_trace(go.Scatter3d(
                    x=[bx],
                    y=[by],
                    z=[bz],
                    mode="markers",
                    marker=dict(size=12, color="red"),
                    name="Optimal"
                ))
        except Exception:
            # Ne pas planter l'affichage si conversion échoue
            pass

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

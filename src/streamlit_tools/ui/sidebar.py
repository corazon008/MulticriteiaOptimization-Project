import streamlit as st


def render_sidebar(mu):
    st.sidebar.header("🎯 Paramètres")

    model = st.sidebar.radio(
        "Modèle",
        ("Markowitz", "Contraintes & Coûts")
    )

    r_min = st.sidebar.slider(
        "Rendement minimal",
        min_value=float(max(0, mu.min())),
        max_value=float(mu.max()),
        value=float(mu.mean()),
        format="%.4f"
    )

    K, c = None, None
    if model == "Contraintes & Coûts":
        K = st.sidebar.number_input("Cardinalité K", 2, len(mu), 5)
        c = st.sidebar.number_input("Coût de transaction (%)", 0.0, value=0.01)

    return model, r_min, K, c

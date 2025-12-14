import streamlit as st
import numpy as np
import pandas as pd


def init_w0_state(asset_names):
    st.session_state.setdefault("w0_dict", {})
    st.session_state.setdefault("w0_array", None)


def render_home(asset_names):
    init_w0_state(asset_names)
    st.title("Accueil — Définir votre portefeuille initial (W0)")

    col1, col2 = st.columns([2, 3])

    with col1:
        asset = st.selectbox("Choisir un actif (rechercher par nom)", options=asset_names)
        weight = st.number_input("Pondération (par rapport à 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
        if st.button("Ajouter / Mettre à jour"):
            st.session_state["w0_dict"][asset] = float(weight)
        if st.button("Supprimer l'actif sélectionné"):
            st.session_state["w0_dict"].pop(asset, None)
        st.markdown("### Options")
        normalize = st.checkbox("Normaliser les poids pour que la somme soit 1 lors de l'enregistrement", value=True)
        if st.button("Enregistrer W0 et continuer"):
            arr = np.zeros(len(asset_names), dtype=float)
            for i, a in enumerate(asset_names):
                arr[i] = float(st.session_state["w0_dict"].get(a, 0.0))
            if normalize:
                s = arr.sum()
                if s > 0:
                    arr = arr / s
            st.session_state["w0_array"] = arr
            st.success("W0 enregistré dans session_state['w0_array']. Vous pouvez maintenant aller aux modèles.")
            st.session_state["page"] = "Modèles"

    with col2:
        st.markdown("### Aperçu des pondérations ajoutées")
        if st.session_state["w0_dict"]:
            df = pd.DataFrame.from_dict(st.session_state["w0_dict"], orient="index", columns=["weight"])
            df.index.name = "asset"
            st.dataframe(df.sort_values("weight", ascending=False))
            total = sum(st.session_state["w0_dict"].values())
            st.write(f"Somme actuelle des poids (non normalisée): {total:.6f}")
        else:
            st.info("Aucun actif ajouté. Utilisez la colonne de gauche pour ajouter des actifs et leurs pondérations.")

    st.markdown("---")
    st.write("Session keys disponibles :", list(st.session_state.keys()))


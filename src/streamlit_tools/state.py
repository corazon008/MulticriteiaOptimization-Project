import streamlit as st


def init_state():
    st.session_state.setdefault("model", "Markowitz")
    # Page courante (Accueil, Paramètres, Modèles)
    st.session_state.setdefault("page", "Accueil")
    # Portefeuille initial W0 : stocké comme dict asset->poids puis array dans 'w0_array'
    st.session_state.setdefault("w0_dict", {})
    st.session_state.setdefault("w0_array", None)
    # Paramètres NSGA par défaut
    st.session_state.setdefault("nsga_pop", 100)
    st.session_state.setdefault("nsga_gen", 200)
    st.session_state.setdefault("lambda_count", 50)
    st.session_state.setdefault("nsga_mutation", 0.05)

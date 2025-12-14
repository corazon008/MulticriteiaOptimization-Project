import streamlit as st


def render_settings():
    st.title("Paramètres")
    st.markdown("Paramètres pour NSGA2 et génération de lambdas")

    # valeurs par défaut si non présentes
    st.session_state.setdefault("nsga_pop", 100)
    st.session_state.setdefault("nsga_gen", 200)
    st.session_state.setdefault("lambda_count", 50)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["nsga_pop"] = st.number_input("Population NSGA2", min_value=10, max_value=5000, step=10, value=int(st.session_state["nsga_pop"]))
        st.session_state["nsga_gen"] = st.number_input("Générations NSGA2", min_value=1, max_value=5000, step=1, value=int(st.session_state["nsga_gen"]))
    with col2:
        st.session_state["lambda_count"] = st.number_input("Nombre de lambda (linspace)", min_value=2, max_value=2000, step=1, value=int(st.session_state["lambda_count"]))
        mutation = st.slider("Taux de mutation (optionnel)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        st.session_state["nsga_mutation"] = mutation

    if st.button("Sauvegarder les paramètres"):
        st.success("Paramètres sauvegardés dans session_state.")
    st.write("Session keys disponibles :", list(st.session_state.keys()))


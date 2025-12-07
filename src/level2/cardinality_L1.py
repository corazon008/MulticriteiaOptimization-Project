import cvxpy as cp
import numpy as np

from src.portfolio_utils import f_rendement, f_risque

def optimize(mu: np.ndarray, Sigma: np.ndarray,  K: int, epsilons: np.ndarray, lambda_penalty: float) -> tuple[list[float], list[float], list[np.ndarray]]:
    n = mu.shape[0]

    # ----------------------------
    # Résultats
    # ----------------------------
    frontier_returns = []
    frontier_volatilities = []
    frontier_weights = []

    # ----------------------------
    # Boucle ε-contrainte
    # ----------------------------
    for eps in epsilons:

        w = cp.Variable(n)

        # Objectif : maximise le rendement avec pénalisation L1 pour sparsité
        objective = cp.Minimize(-mu @ w + lambda_penalty * cp.sum(w))

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            cp.quad_form(w, Sigma) <= eps
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Aucune solution pour eps = {eps}")
            continue

        w_val = w.value

        # ----------------------------
        # Post-traitement : sélectionner les K plus gros actifs
        # ----------------------------
        top_indices = np.argsort(-w_val)[:K]  # indices des K plus gros
        w_sparse = np.zeros_like(w_val)
        w_sparse[top_indices] = w_val[top_indices]
        w_sparse /= w_sparse.sum()  # renormaliser pour que sum=1

        # Stockage des résultats
        ret_val = f_rendement(w_sparse, mu)
        risk_val = np.sqrt(f_risque(w_sparse, Sigma))

        frontier_returns.append(ret_val)
        frontier_volatilities.append(risk_val)
        frontier_weights.append(w_sparse)

    return frontier_returns, frontier_volatilities, frontier_weights
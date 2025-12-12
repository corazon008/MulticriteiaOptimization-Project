import cvxpy as cp
import numpy as np

from portfolio_utils import f_yield, f_volatility
from level2.functions import nb_not_null_weights

def optimize(mu: np.ndarray, Sigma: np.ndarray,  K: int, epsilons: np.ndarray, lambda_penalty: float) -> tuple[list[float], list[float], list[np.ndarray]]:
    n = mu.shape[0]

    # ----------------------------
    # Résultats
    # ----------------------------
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

        frontier_weights.append(w_sparse)

    frontier_weights = np.array([w for w in frontier_weights if nb_not_null_weights(w, 1e-4) == K])

    frontier_yields = np.array([f_yield(w, mu) for w in frontier_weights])
    frontier_volatility = np.array([f_volatility(w, Sigma) for w in frontier_weights])

    return frontier_yields, frontier_volatility, frontier_weights
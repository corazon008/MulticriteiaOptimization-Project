from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from tqdm import tqdm
import math
import pandas as pd
import cvxpy as cp
import numpy as np

from level2.functions import nb_not_null_weights
from portfolio_utils import f_volatility, f_yield


def optimize(mu: np.ndarray, Sigma: np.ndarray,  K: int, epsilons: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:

    n = mu.shape[0]

    # ----------------------------
    # Variables du modèle
    # ----------------------------
    w = cp.Variable(n)  # pondérations du portefeuille
    z = cp.Variable(n, boolean=True)  # variables binaires pour la cardinalité

    # Résultats stockés
    frontier_weights = []

    for eps in epsilons:

        # ----------------------------
        # Problème ε-contraint
        # ----------------------------
        objective = cp.Minimize(-mu @ w)  # Minimiser F1 = -w^T mu (maximiser le rendement)

        constraints = [
            cp.sum(w) == 1,  # Somme des poids = 1
            w >= 0,  # Long-only (à adapter)
            w <= z,  # Lie w et z (si z=0 => w=0)
            cp.sum(z) <= K,  # Cardinalité
            cp.quad_form(w, Sigma) <= eps,  # Risque F2 <= epsilon
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCIP, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Pas de solution pour epsilon = {eps}")
            continue

        # Stockage des solutions
        w_val = w.value
        ret_val = f_yield(w_val, mu)
        risk_val = f_volatility(w_val, Sigma)

        frontier_weights.append(w_val)

    frontier_weights = np.array([w for w in frontier_weights if nb_not_null_weights(w, 1e-4) == K])

    frontier_yields = np.array([f_yield(w, mu) for w in frontier_weights])
    frontier_volatility = np.array([f_volatility(w, Sigma) for w in frontier_weights])

    return frontier_yields, frontier_volatility, frontier_weights
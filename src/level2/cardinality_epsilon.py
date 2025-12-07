from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from tqdm import tqdm
import math
import pandas as pd
import cvxpy as cp
import numpy as np


def optimize(mu: np.ndarray, Sigma: np.ndarray,  K: int, epsilons: np.ndarray) -> tuple[list[float], list[float], list[np.ndarray]]:

    n = mu.shape[0]


    # ----------------------------
    # Variables du modèle
    # ----------------------------
    w = cp.Variable(n)  # pondérations du portefeuille
    z = cp.Variable(n, boolean=True)  # variables binaires pour la cardinalité

    # Résultats stockés
    frontier_returns = []
    frontier_risks = []
    weights = []

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
        problem.solve(solver=cp.ECOS_BB, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Pas de solution pour epsilon = {eps}")
            continue

        # Stockage des solutions
        w_val = w.value
        ret_val = float(mu @ w_val)
        risk_val = float(w_val.T @ Sigma @ w_val)

        frontier_returns.append(ret_val)
        frontier_risks.append(risk_val)
        weights.append(w_val)

    return np.array(frontier_returns), np.array(frontier_risks), weights
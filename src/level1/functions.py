import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_utils import *


def f_objective(w, lambda_param, mu, Sigma):
    portfolio_return = f_rendement(w, mu)
    portfolio_variance = f_risque(w, Sigma)
    return lambda_param * portfolio_variance - (1 - lambda_param) * portfolio_return


def optimize_portfolio(lambdas, mu: np.ndarray, Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    num_assets = len(mu)

    # Contraintes
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum w_i = 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # w_i entre 0 et 1

    # Point de départ (portefeuille équipondéré)
    initial_guess = np.array(num_assets * [1. / num_assets])

    # Générer la frontière efficiente en variant lambda
    frontier_returns = []
    frontier_volatilities = []
    frontier_weights = []

    for lambda_param in lambdas:
        result = minimize(f_objective, initial_guess, args=(lambda_param, mu, Sigma), method='SLSQP', bounds=bounds,
                          constraints=constraints)
        if result.success:
            w_opt = result.x
            frontier_returns.append(f_rendement(w_opt, mu))
            frontier_volatilities.append(np.sqrt(f_risque(w_opt, Sigma)))
            frontier_weights.append(w_opt)
        else:
            print(f"Optimization failed for lambda={lambda_param}")

    # Convertir en arrays numpy pour faciliter l'analyse
    frontier_returns = np.array(frontier_returns)
    frontier_volatilities = np.array(frontier_volatilities)

    return frontier_returns, frontier_volatilities, frontier_weights
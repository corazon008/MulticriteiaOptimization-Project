# pareto_scalar.py
import numpy as np
from portfolio_utils import solve_markowitz, compute_portfolio_stats

def pareto_scalar(mu, Sigma, n_points=50):
    mus = []
    vols = []
    weights = []
    # Use target returns between min and max single-asset returns
    min_r = np.min(mu)
    max_r = np.max(mu)
    targets = np.linspace(min_r, max_r, n_points)
    for t in targets:
        try:
            w = solve_markowitz(mu, Sigma, target_return=float(t))
            stats = compute_portfolio_stats(w, mu, Sigma)
            mus.append(stats["return"])
            vols.append(stats["vol"])
            weights.append(w)
        except Exception as e:
            # skip infeasible targets
            continue
    return np.array(mus), np.array(vols), weights

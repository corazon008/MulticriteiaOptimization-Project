import numpy as np
import pandas as pd

def annualize_returns(daily_returns, periods_per_year=252):
    mu = daily_returns.mean() * periods_per_year
    return mu

def annualize_cov(daily_returns, periods_per_year=252):
    return daily_returns.cov() * periods_per_year

def random_synthetic_data(n_assets=10, n_days=252*3, seed=42):
    np.random.seed(seed)
    # simulate correlated returns
    A = np.random.randn(n_assets, n_assets)
    cov = np.dot(A, A.T)
    mean = np.random.uniform(0.02, 0.15, size=n_assets) / 252  # daily means
    returns = np.random.multivariate_normal(mean*np.ones(n_assets), cov/1000.0, size=n_days)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
    return pd.DataFrame(returns, index=dates, columns=[f'Asset{i+1}' for i in range(n_assets)])

def solve_markowitz(mu, Sigma, target_return=None):
    # minimize w^T Sigma w  subject to sum(w)=1, w>=0, and optional w^T mu >= target_return
    n = len(mu)
    w = cp.Variable(n)
    ret = mu.values if hasattr(mu, "values") else mu
    Sigma_np = Sigma.values if hasattr(Sigma, "values") else Sigma
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma_np)),
                      [cp.sum(w) == 1,
                       w >= 0])
    if target_return is not None:
        prob.constraints += [w @ ret >= target_return]
    prob.solve(solver=cp.OSQP)  # OSQP works for QP
    return np.array(w.value).flatten()

def compute_portfolio_stats(w, mu, Sigma):
    port_ret = float(np.dot(w, mu))
    port_var = float(w @ Sigma @ w)
    port_std = np.sqrt(port_var)
    return {"return": port_ret, "vol": port_std, "var": port_var}

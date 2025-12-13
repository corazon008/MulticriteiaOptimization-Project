import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair

from portfolio_utils import f_yield, f_volatility, f_cost, load_datas, f_returns_on_df, f_mu_on_df, f_sigma_on_df

def nb_not_null_weights(weights: np.ndarray, threshold: float = 1e-6) -> int:
    """Compte le nombre de poids non nuls dans un vecteur de poids."""
    return np.sum(weights > threshold)

class PortfolioNSGA2(ElementwiseProblem):

    def __init__(self, mu, Sigma, w0, K, delta_tol, c=0.01):
        self.mu = mu
        self.Sigma = Sigma
        self.w0 = w0
        self.delta_tol = delta_tol
        self.K = K
        self.c = c
        n_assets = len(mu)

        super().__init__(n_var=n_assets,
                         n_obj=3,
                         n_ieq_constr=0,
                         n_eq_constr=2,            # contrainte : somme = 1
                         xl=0.0,
                         xu=1.0)

    def _evaluate(self, w, out, *args, **kwargs):

        f1 = -f_yield(w, self.mu)                       # minimiser → rendement max
        f2 = f_volatility(w, self.Sigma)                # minimiser volatilité
        f3 = f_cost(self.w0, w, self.c)  # minimiser coût

        # Contrainte égalité : somme(w) = 1
        h1 = np.sum(w) - 1
        h2 = np.sum(w > self.delta_tol) - self.K

        out["F"] = [f1, f2, f3]
        out["H"] = [h1, h2]

class CardinalityRepair(Repair):
    def __init__(self, K):
        super().__init__()
        self.K = K

    def _do(self, problem, X, **kwargs):
        # X : population, shape (n_individuals, n_assets)
        X = X.copy()

        for i in range(X.shape[0]):
            w = X[i]

            # Trouver les indices des K plus grands poids
            idx = np.argsort(-w)[:self.K]

            # Tout mettre à zéro
            w_new = np.zeros_like(w)

            # Garder seulement les K actifs
            w_new[idx] = w[idx]

            # Renormaliser pour somme=1
            s = np.sum(w_new)
            if s > 0:
                w_new /= s

            X[i] = w_new

        return X


def optimize(mu: pd.Series, Sigma: pd.Series, w0: np.ndarray, K: int, delta_tol, population_size: int = 100, generations: int = 200, c:float=0.01) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    problem = PortfolioNSGA2(mu, Sigma, w0, K, delta_tol=delta_tol, c=c)

    algorithm = NSGA2(
        pop_size=population_size,
        repair=CardinalityRepair(K)
    )

    termination = get_termination("n_gen", generations)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   verbose=True)

    frontier_weights = res.X
    frontier_weights = np.array([w for w in frontier_weights if nb_not_null_weights(w, 1e-4) == K])

    frontier_yields = np.array([f_yield(w, mu) for w in frontier_weights])
    frontier_volatility = np.array([f_volatility(w, Sigma) for w in frontier_weights])
    frontier_cost = np.array([f_cost(w0, w, c) for w in frontier_weights])

    return frontier_yields, frontier_volatility, frontier_cost, frontier_weights

if __name__ == "__main__":
    # Test rapide
    df = load_datas()

    # Calcul des rendements logarithmiques
    returns = f_returns_on_df(df)

    # Calcul des paramètres pour l'optimisation
    mu = f_mu_on_df(returns)  # Annualisation (252 jours boursiers)
    Sigma = f_sigma_on_df(returns)  # Annualisation de la matrice de covariance
    num_assets = len(mu)
    mu = mu.values.astype(float)  # shape (196,)
    Sigma = Sigma.values.astype(float)

    w0 = np.zeros(num_assets)  # Portefeuille initial (vide)
    w0[2] = 1.0  # Tout investir sur NVIDIA par exemple

    K = 3  # Cardinalité
    frontier_yields, frontier_volatility, frontier_cost, frontier_weights = optimize(mu, Sigma, w0, K, population_size=50, generations=200, c=1)

    max_return_index = np.argmax(frontier_yields)
    w_max = frontier_weights[max_return_index]
    print(f"Portefeuille avec rendement maximal (exact {K} actifs) :")
    print(f"Rendement : {frontier_yields[max_return_index]:.4f}")
    print(f"Volatilité : {frontier_volatility[max_return_index]:.4f}")
    print(f"Coût : {frontier_cost[max_return_index]:.4f}")
    for i, weight in enumerate(w_max):
        if weight > 1e-4:
            print(f"  {df.columns[i]} : {weight:.4f}")

    # Min volatilité
    min_vol_index = np.argmin(frontier_volatility)
    w_min = frontier_weights[min_vol_index]
    print(f"\nPortefeuille avec risque minimal (exact {K} actifs) :")
    print(f"Rendement : {frontier_yields[min_vol_index]:.4f}")
    print(f"Volatilité : {frontier_volatility[min_vol_index]:.4f}")
    print(f"Coût : {frontier_cost[min_vol_index]:.4f}")
    for i, weight in enumerate(w_min):
        if weight > 1e-4:
            print(f"  {df.columns[i]} : {weight:.4f}")
import pandas as pd
import numpy as np

from level2.functions import optimize
from portfolio_utils import f_returns_on_df, f_mu_on_df, f_sigma_on_df, f_yield, f_volatility


class PortfolioRobustness:
    def __init__(self, df, w0, K, delta_tol, c=0.01):
        self.df = df
        self.w0 = w0
        self.K = K
        self.delta_tol = delta_tol
        self.c = c

        re = f_returns_on_df(df)
        self.mu = f_mu_on_df(re)
        self.Sigma = f_sigma_on_df(re)

    def optimize(self, population_size=300, generations=100):
        self.frontier_yields, self.frontier_volatility, self.frontier_cost, self.frontier_weights = optimize(self.mu,
                                                                                                             self.Sigma,
                                                                                                             self.w0,
                                                                                                             self.K,
                                                                                                             delta_tol=self.delta_tol,
                                                                                                             population_size=population_size,
                                                                                                             generations=generations,
                                                                                                             c=self.c)

    def boostrap_sample_df(self) -> dict[int, pd.DataFrame]:
        years = sorted(set(self.df.index.year))
        bootstrapped_dfs = {}
        for y in years:
            df_year = self.df[self.df.index.year == y]
            bootstrapped_dfs[y] = df_year
        return bootstrapped_dfs

    def evaluate_portfolio_over_years(self, w):
        year_params = {}
        for year, b in self.boostrap_sample_df().items():
            re_y = f_returns_on_df(b)
            mu_y = f_mu_on_df(re_y)
            Sigma_y = f_sigma_on_df(re_y)
            year_params[year] = (mu_y, Sigma_y)

        rets = []
        vols = []
        for mu_y, Sigma_y in year_params.values():
            r = f_yield(w, mu_y)
            v = f_volatility(w, Sigma_y)
            rets.append(r)
            vols.append(v)
        return np.array(rets), np.array(vols)

    def compute_score(self, vol_per: float, cost_per: float, yield_per: float, yield_std_per: float,
                      vol_std_per: float):
        """
        Calcule un score composite pour chaque portefeuille de la frontière en tenant compte
        des préférences utilisateur.
        5 critères sont pris en compte :
        - niveau de risque (volatilité)
        - coûts de transaction
        - niveau de rendement
        - instabilité du rendement (écart-type des rendements annuels)
        - instabilité du risque (écart-type des volatilités annuelles)
        Les poids relatifs de chaque critère sont définis par les paramètres d'entrée.
        Le score final est une somme pondérée de ces critères normalisés.
        """
        self.std_yields = []
        self.std_vols = []

        for w in self.frontier_weights:
            rets, vols = self.evaluate_portfolio_over_years(w)
            self.std_yields.append(np.std(rets))  # f4 = instabilité rendement
            self.std_vols.append(np.std(vols))

        return (vol_per * normalize(self.frontier_volatility) +  # niveau de risque
                cost_per * normalize(self.frontier_cost) +  # coûts
                -yield_per * normalize(self.frontier_yields) +  # niveau de rendement
                yield_std_per * normalize(self.std_yields) +  # instabilité rendement
                vol_std_per * normalize(self.std_vols)  # instabilité risque
                )


def normalize(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)  # 1e-12 to avoid zero division

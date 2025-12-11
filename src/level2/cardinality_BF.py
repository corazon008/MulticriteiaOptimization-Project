from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from tqdm import tqdm
import math
import pandas as pd

from portfolio_utils import *
from level1.functions import optimize_portfolio

# ---- Variables globales pour partager df/lambdas sans les pickliser 1000 fois ---- #
df_global = None
lambdas_global = None

def init_worker(df, lambdas):
    global df_global, lambdas_global
    df_global = df
    lambdas_global = lambdas

def worker(possibility):
    df = df_global
    lambdas = lambdas_global

    selected_columns = df.columns[list(possibility)]
    temp_df = df[selected_columns]

    returns = f_returns_on_df(temp_df)
    mu = f_mu_on_df(returns)
    Sigma = f_sigma_on_df(returns)

    return optimize_portfolio(lambdas, mu, Sigma)

def optimize(df: pd.DataFrame, number_of_shares: int, lambdas: np.ndarray, max_workers: int = 8):

    num_assets = df.shape[1]
    possibilities = combinations(range(num_assets), number_of_shares)

    # Nombre total de combinaisons (pour tqdm)
    total = math.comb(num_assets, number_of_shares)

    frontier_yield = []
    frontier_volatilities = []
    frontier_weights = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(df, lambdas)
    ) as executor:

        for fr, fv, fw in tqdm(
            executor.map(worker, possibilities, chunksize=10),
            total=total,
            desc="Optimizing"
        ):
            frontier_yield.append(fr)
            frontier_volatilities.append(fv)
            frontier_weights.append(fw)

    return frontier_yield, frontier_volatilities, frontier_weights

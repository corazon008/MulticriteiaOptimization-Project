from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import math
import pandas as pd

from src.portfolio_utils import *
from src.level1.functions import optimize_portfolio


def worker(possibility, df: pd.DataFrame, lambdas: np.ndarray):
    selected_columns = df.columns[list(possibility)]
    temp_df = df[selected_columns]

    returns = f_returns(temp_df)
    mu = f_mu(returns)
    Sigma = f_sigma(returns)

    return optimize_portfolio(lambdas, mu, Sigma)


def optimize(df: pd.DataFrame, number_of_shares: int, lambdas: np.ndarray, max_workers: int = 8) -> tuple[
    list[np.ndarray], list[np.ndarray], list[list[np.ndarray]]]:
    num_assets = df.shape[1]

    possibilities = combinations(range(num_assets), number_of_shares)

    frontier_returns = []
    frontier_volatilities = []
    frontier_weights = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, possibility in enumerate(possibilities):
            futures.append(executor.submit(worker, possibility, df, lambdas))

        # barre de progression sur les futures
        for f in tqdm(as_completed(futures), total=len(futures), desc="Optimizing"):
            fr, fv, fw = f.result()
            frontier_returns.append(fr)
            frontier_volatilities.append(fv)
            frontier_weights.append(fw)

    return frontier_returns, frontier_volatilities, frontier_weights

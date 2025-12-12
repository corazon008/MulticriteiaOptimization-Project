import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_datas() -> pd.DataFrame:
    path = Path(__file__).resolve().parent.parent / 'datasets'
    df = pd.read_csv(path / 'Information_Technology.csv', index_col=0, parse_dates=True)

    for file in os.listdir(path):
        if file.endswith('.csv') and 'Information_Technology' not in file:
            temp_df = pd.read_csv(os.path.join(path, file), index_col=0, parse_dates=True)
            df = df.join(temp_df, how='inner')
    return df

def f_share_stats(df:pd.DataFrame, tick:str):
    returns = f_returns_on_df(df)
    mu = f_mu_on_df(returns)
    sigma = f_sigma_on_df(returns)
    return {'yield': mu[tick], 'volatility': np.sqrt(sigma.loc[tick, tick])}

def f_returns_on_df(df:pd.DataFrame) -> pd.DataFrame:
    return np.log(df / df.shift(1)).dropna()

def f_mu_on_df(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * 252

def f_sigma_on_df(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * 252

def f_yield(w, mu):
    return np.dot(w, mu)

def f_volatility(w, Sigma):
    return np.sqrt(np.dot(w, np.dot(Sigma, w)))

def f_cost(w0, w, transaction_cost_rate:float=0.001):
    return transaction_cost_rate * np.sum(np.abs(w - w0))

if __name__ == "__main__":
    df = load_datas()
    print(f_share_stats(df, "ANET"))
    print(f_share_stats(df, "NVDA"))
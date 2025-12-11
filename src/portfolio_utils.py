import pandas as pd
import numpy as np
import os

def load_datas() -> pd.DataFrame:
    df = pd.read_csv('../datasets/Information_Technology.csv', index_col=0, parse_dates=True)

    for file in os.listdir('../datasets/'):
        if file.endswith('.csv') and 'Information_Technology' not in file:
            temp_df = pd.read_csv(os.path.join('../datasets/', file), index_col=0, parse_dates=True)
            df = df.join(temp_df, how='inner')
    return df

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

def f_cost(w0, w, transaction_cost_rate=0.001):
    return transaction_cost_rate * np.sum(np.abs(w - w0))
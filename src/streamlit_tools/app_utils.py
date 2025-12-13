import pandas as pd
import numpy as np
import os
import glob
import pickle
from scipy.optimize import minimize

from level1 import functions as level1
from level2 import functions as level2
from level3 import functions as level3


def get_ticker_sector_map(dataset_path="datasets"):
    """
    Crée un dictionnaire {Ticker: Secteur} en scannant les noms de fichiers CSV.
    Exemple: Si AAPL est dans 'Information_Technology.csv', alors map['AAPL'] = 'Information_Technology'
    """
    mapping = {}
    # Adaptez le chemin si nécessaire selon où vous lancez le script
    if not os.path.exists(dataset_path):
        # Fallback si lancé depuis la racine ou src
        if os.path.exists(os.path.join("..", dataset_path)):
            dataset_path = os.path.join("..", dataset_path)

    pattern = os.path.join(dataset_path, "*.csv")
    files = glob.glob(pattern)

    for file in files:
        sector_name = os.path.basename(file).replace(".csv", "")
        try:
            # On lit juste la première ligne pour avoir les tickers (colonnes)
            df = pd.read_csv(file, nrows=1, index_col=0)
            tickers = df.columns.tolist()
            for t in tickers:
                mapping[t] = sector_name
        except Exception as e:
            print(f"Warning: Impossible de lire {file} pour le mapping secteur. {e}")

    return mapping


def calculate_markowitz_frontier(mu, Sigma, num_points=30):
    results = []

    lambdas = np.linspace(0, 1, 50)
    frontier_yields, frontier_volatility, frontier_weights = level1.optimize_portfolio(lambdas, mu, Sigma)

    for r, v, w in zip(frontier_yields, frontier_volatility, frontier_weights):
        results.append({
            'return': r,
            'volatility': v,
            'weights': w,
            'sharpe': r / v if v > 0 else 0
        })

    return pd.DataFrame(results)

def calculate_portfolio(mu, Sigma, w0, K, c):
    results = []

    frontier_yields, frontier_volatility, frontier_cost, frontier_weights = level2.optimize(mu=mu, Sigma=Sigma, w0=w0, population_size=200, generations=300,delta_tol=0.01, K=K, c=c)

    for r, v, cost, w in zip(frontier_yields, frontier_volatility, frontier_cost, frontier_weights):
        results.append({
            'return': r,
            'volatility': v,
            'cost': cost,
            'weights': w,
            'sharpe': r / v if v > 0 else 0
        })

    return pd.DataFrame(results)
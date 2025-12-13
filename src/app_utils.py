import pandas as pd
import numpy as np
import os
import glob
import pickle
from scipy.optimize import minimize


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


def load_saved_frontier(filepath):
    """
    Charge les résultats sauvegardés dans les notebooks (fichiers .pkl).
    Gère les formats list de dicts ou dict de dicts.
    """
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Conversion en DataFrame robuste
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient='index')
        else:
            df = pd.DataFrame(data)

        # Standardisation des noms de colonnes pour l'app
        # Vos notebooks peuvent avoir utilisé 'risk' ou 'std', on veut 'volatility'
        rename_map = {'risk': 'volatility', 'std': 'volatility', 'yield': 'return', 'mu': 'return'}
        df = df.rename(columns=rename_map)

        return df
    except Exception as e:
        print(f"Erreur chargement pickle {filepath}: {e}")
        return None


def calculate_markowitz_frontier(mu, Sigma, num_points=30):
    """
    Calcule la frontière efficiente classique (Markowitz) à la volée.
    Utile si le fichier pickle n'est pas dispo ou pour comparaison.
    """
    results = []
    # On balaie du rendement min au rendement max
    target_returns = np.linspace(mu.min(), mu.max(), num_points)

    n = len(mu)
    # Contrainte de poids entre 0 et 1 (pas de vente à découvert)
    bounds = tuple((0, 1) for _ in range(n))

    for r in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Somme = 1
            {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - r}  # Rendement = r
        )

        # Minimisation de la variance
        res = minimize(lambda x: np.dot(x, np.dot(Sigma, x)),
                       n * [1. / n],  # Poids initiaux équilibrés
                       method='SLSQP', bounds=bounds, constraints=constraints)

        if res.success:
            vol = np.sqrt(res.fun)
            results.append({
                'return': r,
                'volatility': vol,
                'weights': res.x,
                'sharpe': r / vol if vol > 0 else 0
            })

    return pd.DataFrame(results)
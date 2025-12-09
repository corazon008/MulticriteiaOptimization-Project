from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import math

import portfolio_utils
from level2.cardinality_epsilon import optimize
from level1.functions import *


if __name__ == "__main__":
    df = portfolio_utils.load_datas()
    returns = f_returns(df)
    mu = f_mu(returns).to_numpy()
    Sigma = f_sigma(returns).to_numpy()
    K = 3  # Nombre d'actifs à sélectionner
    epsilons = np.linspace(0.0001, 0.01, 10)
    fr, fv, fw = optimize(mu, Sigma, K, epsilons)
    print("Rendements optimaux :", fr)
    print("Risques optimaux :", fv)
    print("Poids optimaux :", fw)

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import math

import portfolio_utils
from level2.cardinality_epsilon import optimize
from src.level1.functions import *





if __name__ == "__main__":
    df = portfolio_utils.load_datas()
    returns = f_returns(df)
    mu = f_mu(returns).to_numpy()
    Sigma = f_sigma(returns).to_numpy()
    fr, fv, fw = optimize(mu, Sigma)
    print("Rendements optimaux :", fr)
    print("Risques optimaux :", fv)
    print("Poids optimaux :", fw)

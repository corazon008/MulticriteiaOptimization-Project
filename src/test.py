from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import math

import portfolio_utils
from level1.functions import  *

df = portfolio_utils.load_datas()

lambdas = np.linspace(0, 1, 50)
num_assets = df.shape[1]
number_of_shares = 2

if __name__ == "__main__":
    from src.level2.cardinality_BF import optimize

    fr, fv, fw = optimize(df, 2)

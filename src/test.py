from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from tqdm import tqdm
import math

import portfolio_utils
from level2.cardinality_epsilon import optimize
from level1.functions import *
from portfolio_utils import f_share_stats


if __name__ == "__main__":
    df = portfolio_utils.load_datas()
    print(f_share_stats(df, "ANET"))
    print(f_share_stats(df, "NVDA"))

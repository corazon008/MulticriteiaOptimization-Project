import pandas as pd

def boostrap_sample_df(df:pd.DataFrame)-> dict[int, pd.DataFrame]:
    years = sorted(set(df.index.year))
    bootstrapped_dfs = {}
    for y in years:
        df_year = df[df.index.year == y]
        bootstrapped_dfs[y] = df_year
    return bootstrapped_dfs
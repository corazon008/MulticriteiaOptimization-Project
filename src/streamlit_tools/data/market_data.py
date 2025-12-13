import streamlit as st
from portfolio_utils import load_datas, f_mu_on_df, f_sigma_on_df, f_returns_on_df
from streamlit_tools.app_utils import get_ticker_sector_map


@st.cache_data
def load_market_data():
    df_prices = load_datas()
    returns = f_returns_on_df(df_prices)
    mu = f_mu_on_df(returns)
    sigma = f_sigma_on_df(returns)
    sector_map = get_ticker_sector_map("datasets")
    return df_prices, mu, sigma, sector_map

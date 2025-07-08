import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def download_prices(ticker1, ticker2, start="2020-01-01", end="2025-7-08"):
    # downlads adjusted close prices for two tickers using yfinance
    # handles multi-level columns for multiple tickers.

    data = yf.download([ticker1, ticker2], start=start, end=end, group_by='ticker', auto_adjust=True)
    
    # extract individual series safely
    if ticker1 not in data or ticker2 not in data:
        raise ValueError(f"Failed to retrieve data for {ticker1} or {ticker2}")
        
    y = data[ticker1]['Close'].dropna()
    x = data[ticker2]['Close'].dropna()
    
    # aligns the two series by date
    combined = pd.concat([y, x], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


def engle_granger_test(y, x, adf_alpha=0.05):

    # performs engle-granger cointegration test
    # returns hedge ratio, p-value, and cointegration status.

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    hedge_ratio = model.params[1]
    spread = y - model.predict(x)

    # checking if the series is mean-reverting
    adf_stat, pvalue, _, _, _, _ = adfuller(spread)
    cointegrated = pvalue < adf_alpha

    return {
        "hedge_ratio": hedge_ratio,
        "adf_pvalue": pvalue,
        "cointegrated": cointegrated,
        "spread_series": spread,
        "adf_stat": adf_stat,
        "y": y,
        "x": x.iloc[:, 1] 
    }

if __name__ == "__main__":
    # put ur stocks here
    y, x = download_prices("JNJ", "PG")
    result = engle_granger_test(y, x)

    print("Cointegrated:", result["cointegrated"])
    print("ADF p-value:", result["adf_pvalue"])
    # replace the names with chosen stock tickers
    print("Hedge Ratio (JNJ/PG):", result["hedge_ratio"])

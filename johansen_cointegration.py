import yfinance as yf  # Yahoo Finance data downloader
import pandas as pd       # Data manipulation
import numpy as np        # Numerical operations
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # Johansen cointegration


def download_prices(
    ticker1: str,
    ticker2: str,
    start: str = "2020-01-01",
    end: str | None = None
) -> tuple[pd.Series, pd.Series]:
    """
    Download adjusted close prices for two tickers and align them by date.
    """
    data = yf.download(
        [ticker1, ticker2], start=start, end=end,
        group_by="ticker", auto_adjust=True, progress=False
    )
    levels = getattr(data.columns, 'levels', None)
    if not levels or ticker1 not in levels[0] or ticker2 not in levels[0]:
        raise ValueError(f"Failed to retrieve data for {ticker1} or {ticker2}.")
    y = data[ticker1]["Close"].dropna()
    x = data[ticker2]["Close"].dropna()
    combined = pd.concat([y, x], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


def johansen_test(
    y: pd.Series,
    x: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 1,
    alpha: float = 0.05
) -> tuple[bool, float, float, float]:
    """
    Run Johansen cointegration test.
    """
    df = pd.concat([y, x], axis=1)
    df.columns = ["y", "x"]
    result = coint_johansen(df, det_order, k_ar_diff)
    alpha_map = {0.10: 0, 0.05: 1, 0.01: 2}
    idx = alpha_map.get(alpha, 1)
    trace_stat = result.lr1[0]
    crit_val = result.cvt[0, idx]
    cointegrated = trace_stat > crit_val
    vec = result.evec[:, 0]
    hedge_ratio = -vec[0] / vec[1]
    return cointegrated, trace_stat, crit_val, hedge_ratio


if __name__ == "__main__":
    # --- Configuration: Change tickers and dates below ---
    TICKER1 = "JNJ"   # ← Set your first stock symbol here
    TICKER2 = "PG"     # ← Set your second stock symbol here
    START_DATE = "2020-01-01"
    END_DATE = None      # None => use today's date

    SIGNIFICANCE = 0.05  # alpha for test
    LAGS = 1             # VECM lag order
    DETREND = 0          # deterministic term setting

    # Download data using the ticker variables
    y_series, x_series = download_prices(TICKER1, TICKER2, START_DATE, END_DATE)

    # Perform the Johansen test
    cointegrated, trace_stat, crit_val, hr = johansen_test(
        y_series, x_series,
        det_order=DETREND,
        k_ar_diff=LAGS,
        alpha=SIGNIFICANCE
    )

    # Print results referencing the ticker variables
    print(f"Cointegrated: {cointegrated}")
    print(f"Trace statistic: {trace_stat:.4f}")
    print(f"Critical value @ {SIGNIFICANCE*100:.0f}%: {crit_val:.4f}")
    print(f"Hedge Ratio ({TICKER1}/{TICKER2}): {hr:.4f}")

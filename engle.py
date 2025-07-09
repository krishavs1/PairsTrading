import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from tickers import tickers

# ------------ configurate ------------
START_DATE = "2020-01-01"
END_DATE = "2025-07-08"
ADF_ALPHA = 0.01
MIN_PERIODS = 200
pd.set_option('display.max_rows', None)      # show all rows
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', 200)          # widen the display
# --------------------------------


def download_prices(t1, t2, start, end):
    """
    download and align closing prices for two tickers
    """
    data = yf.download([t1, t2], start=start, end=end,
                       auto_adjust=True, progress=False, group_by='ticker')

    try:
        y = data[t1]["Close"].dropna()
        x = data[t2]["Close"].dropna()
    except Exception:
        try:
            close_df = data.xs('Close', axis=1, level=0).dropna()
            y = close_df[t1]
            x = close_df[t2]
        except Exception:
            return None, None

    combined = pd.concat([y, x], axis=1).dropna()
    if combined.shape[0] < MIN_PERIODS:
        return None, None
    return combined.iloc[:, 0], combined.iloc[:, 1]


def engle_granger_test(y, x, adf_alpha=ADF_ALPHA):
    """
    Perform Engle-Granger cointegration test on two series.
    Returns (cointegrated_flag, p-value, hedge_ratio).
    """
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    hedge_ratio = model.params[1]
    spread = y - model.predict(x_const)
    adf_stat, pvalue, *_ = adfuller(spread)
    return pvalue < adf_alpha, pvalue, hedge_ratio


# results for each pair
results = []
for t1, t2 in combinations(tickers, 2):
    y, x = download_prices(t1, t2, START_DATE, END_DATE)
    if y is None or x is None:
        print(f"[SKIP] {t1}/{t2}: insufficient data or too few periods.")
        continue
    print(f"[TESTING] {t1}/{t2} — {len(y)} points")
    try:
        coint, pval, hr = engle_granger_test(y, x)
        results.append({
            "Ticker 1": t1,
            "Ticker 2": t2,
            "Cointegrated": coint,
            "ADF p-value": round(pval, 5),
            "Hedge Ratio": round(hr, 4)
        })
    except Exception as e:
        print(f"[ERROR] Testing {t1}/{t2} → {e}")

# Build DataFrame
cf = pd.DataFrame(results)

if cf.empty:
    print("No pairs passed")
else:
    print("\nTop Cointegrated Pairs (based on ADF):")
    top = cf[cf["Cointegrated"]].sort_values("ADF p-value")
    print(top.to_string(index=False))

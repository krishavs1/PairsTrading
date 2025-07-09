import yfinance as yf
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from tickers import tickers

# ------------ CONFIG ------------

START_DATE = "2020-01-01"
SIGNIFICANCE = 0.05
LAGS = 1
DETREND = 0
pd.set_option('display.max_rows', None)      # show all rows
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', 200)          # widen the display
# --------------------------------

def download_pair(t1, t2, start):
    data = yf.download([t1, t2], start=start, auto_adjust=True, progress=False)
    try:
        y = data[("Close", t1)].dropna()
        x = data[("Close", t2)].dropna()
    except KeyError:
        return None, None
    combined = pd.concat([y, x], axis=1).dropna()
    if combined.empty or combined.shape[0] < 200:
        return None, None
    return combined.iloc[:, 0], combined.iloc[:, 1]

def johansen_test(y, x, det_order=0, k_ar_diff=1, alpha=0.05):
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

results = []

for t1, t2 in combinations(tickers, 2):
    y, x = download_pair(t1, t2, START_DATE)
    if y is None or x is None:
        print(f"[SKIP] {t1}/{t2}: insufficient data.")
        continue
    print(f"[TESTING] {t1}/{t2} — {len(y)} points")
    try:
        c, stat, crit, hr = johansen_test(y, x, DETREND, LAGS, SIGNIFICANCE)
        results.append({
            "Ticker 1": t1,
            "Ticker 2": t2,
            "Cointegrated": c,
            "Trace Stat": round(stat, 4),
            "Critical Value": round(crit, 4),
            "Hedge Ratio": round(hr, 4)
        })
    except Exception as e:
        print(f"[ERROR] {t1}/{t2}: {e}")

df_results = pd.DataFrame(results)

if df_results.empty:
    print("\n❌ No valid results. Check your tickers or data.")
else:
    print("\n✅ Cointegration Results:")
    print(df_results.sort_values("Trace Stat", ascending=False))

    print("\n📌 Top Cointegrated Pairs:")
    print(df_results[df_results["Cointegrated"] == True].sort_values("Trace Stat", ascending=False))

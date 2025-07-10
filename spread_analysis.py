import yfinance as yf
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import numpy as np

# ------------ CONFIG ------------
T1 = "LLY"                                 
T2 = "AMGN"
START_DATE = "2024-01-01"                   
END_DATE = datetime.today().strftime("%Y-%m-%d")

pd.set_option('display.max_rows', None)      # show all rows
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', 200)          # widen the display
# ---------------------------------------

def download_pair(t1, t2, start, end):
    """
    Download and align adjusted close prices for two tickers.
    Uses auto_adjust + group_by just like your other scripts.
    """
    data = yf.download(
        [t1, t2],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by='ticker'
    )

    try:
        y = data[t1]["Close"].dropna()
        x = data[t2]["Close"].dropna()
    except Exception:
        try:
            # fallback if grouped differently
            close_df = data.xs('Close', axis=1, level=0).dropna()
            y = close_df[t1]
            x = close_df[t2]
        except Exception:
            return None, None

    combined = pd.concat([y, x], axis=1).dropna()
    if combined.empty:
        return None, None

    return combined.iloc[:, 0], combined.iloc[:, 1]

def estimate_hedge_ratio(y, x):
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    return model.params[1]

def compute_spread_metrics(y, x, hr):
    spread = y - hr * x
    mean = spread.mean()
    vol  = spread.std()

    # half-life via Δs_t = α + β·s_{t-1} + ε → hl = -ln(2)/β
    lagged = spread.shift(1).dropna()
    delta  = spread.loc[lagged.index] - lagged
    mdl    = sm.OLS(delta, sm.add_constant(lagged)).fit()
    beta   = mdl.params[1]
    hl     = -np.log(2) / beta if beta < 0 else np.nan

    return spread, mean, vol, hl

if __name__ == "__main__":
    print(f"[RUNNING] Pair: {T1}/{T2}  ({START_DATE} → {END_DATE})")
    y, x = download_pair(T1, T2, START_DATE, END_DATE)
    if y is None or x is None:
        print(f"[ERROR] insufficient data for {T1}/{T2}")
        exit(1)

    hr, (spread, mean, vol, hl) = estimate_hedge_ratio(y, x), compute_spread_metrics(y, x, estimate_hedge_ratio(y, x))

    df = pd.DataFrame([{
        "Pair":           f"{T1}/{T2}",
        "Hedge Ratio":    round(hr,     4),
        "Spread Mean":    round(mean,   4),
        "Spread Vol":     round(vol,    4),
        "Half-Life (d)":  round(hl,     1)
    }])

    print("\nSpread Metrics:")
    print(df.to_string(index=False))

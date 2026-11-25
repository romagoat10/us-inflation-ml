"""Can machine learning forecast US inflation better than traditional models?

Data: a monthly panel of 21 US series (prices, labour market, interest rates, money,
production, exchange rates, oil and gas) downloaded from DBnomics by fetch_data.py.
It is a smaller version of FRED-MD (McCracken and Ng, 2016), which was not reachable.

Target: 12-month CPI inflation h months ahead, for h = 1, 3, 6 and 12.

Models (one fitted directly for each horizon):
    RW     random walk: tomorrow's 12-month inflation equals today's (Atkeson and Ohanian, 2001)
    AR     linear regression on current 12-month inflation and the last 3 monthly rates
    FAAR   AR plus 5 principal-component factors from the whole panel
           (the "diffusion index" model of Stock and Watson, 2002)
    GB     gradient boosted trees on the last 12 monthly inflation rates
    GB+X   GB plus the named drivers: oil and natural gas prices, food prices, the
           federal funds rate, M2 money supply and the dollar exchange rates
    GB+F   GB plus the 5 factors
    RF+F   random forest on the same inputs as GB+F
    AVG    simple average of AR and GB+F

Testing: pseudo out-of-sample with an expanding window. At each forecast date the factors,
scaling and models use only data up to that date. Models are refit every 3 months.
Forecasts start in January 2000.

Accuracy: RMSE relative to RW, for target months 2000-2007, 2008-2019 and 2020 onward,
with Diebold-Mariano tests (Newey-West variance) against RW.

    python forecast.py            -> results/change/  (main results)
    set US_DATA=fredmd first to use the full FRED-MD file data/2026-10-MD.csv
                                  -> results/fredmd_change/
    python forecast.py levels     -> results/levels/  (trees predict the level; first run)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import math
import os
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "us_panel.csv"
TCODES = ROOT / "data" / "us_panel_tcodes.csv"
# Dataset: "dbnomics" (21-series panel, exploratory) or "fredmd" (data/current.csv, confirmatory).
DATASET = os.environ.get("US_DATA", "dbnomics")
assert DATASET in ("dbnomics", "fredmd")
FREDMD = ROOT / "data" / "2026-10-MD.csv"   # FRED-MD vintage downloaded 2026-11-25
# "change": tree models predict the change from today's inflation (main results).
# "levels": tree models predict the level directly (first run; kept for comparison).
TARGET = sys.argv[1] if len(sys.argv) > 1 else "change"
assert TARGET in ("change", "levels")
OUT = ROOT / "results" / (TARGET if DATASET == "dbnomics" else f"fredmd_{TARGET}")
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 3, 6, 12)
START = pd.Timestamp("2000-01-01")     # first forecast origin
REFIT_EVERY = 3                        # months
N_FACTORS = 5
TRAIN_FROM = pd.Timestamp("1973-01-01" if DATASET == "dbnomics" else "1962-01-01")
PERIODS = [("2000-2007", "2000-01-01", "2007-12-01"),
           ("2008-2019", "2008-01-01", "2019-12-01"),
           ("2020-",     "2020-01-01", "2100-01-01")]
NAMED_X = ["OIL", "GAS", "CPI_FOOD", "FEDFUNDS", "M2", "USD_GBP", "JPY_USD", "CAD_USD"]
SEED = 0


# ---------------------------------------------------------------- data
# FRED-MD names for the named drivers (FRED-MD has no food CPI or natural gas price)
FREDMD_NAMES = {"CPIAUCSL": "CPI", "OILPRICEx": "OIL", "M2SL": "M2",
                "EXUSUKx": "USD_GBP", "EXJPUSx": "JPY_USD", "EXCAUSx": "CAD_USD"}


def load_fred_md() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(FREDMD)
    first = raw.columns[0]
    is_t = raw[first].astype(str).str.contains("Transform", case=False)
    tcodes = raw[is_t].iloc[0, 1:].astype(float).astype(int)
    df = raw[~is_t & raw[first].notna()].copy()
    df.index = pd.to_datetime(df[first], format="%m/%d/%Y")
    df = df.drop(columns=first).apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns=FREDMD_NAMES)
    tcodes = tcodes.rename(index=FREDMD_NAMES)
    return df, tcodes


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    if DATASET == "fredmd":
        df, tcodes = load_fred_md()
    else:
        df = pd.read_csv(DATA, index_col=0, parse_dates=True)
        tcodes = pd.read_csv(TCODES, index_col=0)["tcode"].astype(int)
    last_cpi = df["CPI"].last_valid_index()
    return df.loc[:last_cpi], tcodes


def transform(x: pd.Series, code: int) -> pd.Series:
    """FRED-MD stationarity codes (McCracken and Ng, 2016, Table 1)."""
    if code == 1:
        return x
    if code == 2:
        return x.diff()
    if code == 3:
        return x.diff().diff()
    lx = np.log(x.where(x > 0))
    if code == 4:
        return lx
    if code == 5:
        return lx.diff()
    if code == 6:
        return lx.diff().diff()
    if code == 7:
        return (x / x.shift(1) - 1).diff()
    raise ValueError(code)


def build(df: pd.DataFrame, tcodes: pd.Series):
    cpi = df["CPI"]
    pi12 = 100 * (cpi / cpi.shift(12) - 1)          # 12-month inflation, %
    pim = 100 * (cpi / cpi.shift(1) - 1)            # monthly inflation, %
    panel = pd.DataFrame({c: transform(df[c], tcodes[c]) for c in df.columns})
    lags = pd.DataFrame({f"pim_l{k}": pim.shift(k) for k in range(12)})
    ar = pd.DataFrame({"pi12": pi12, **{f"pim_l{k}": pim.shift(k) for k in range(3)}})
    named = pd.DataFrame({f"{c}_d{k}": panel[c].rolling(k).sum()
                          for c in NAMED_X if c in panel for k in (1, 3, 12)})
    return pi12, panel, lags, ar, named


# ---------------------------------------------------------------- factors
def factors(panel: pd.DataFrame, upto: pd.Timestamp, k: int) -> pd.DataFrame:
    """Principal components estimated on data up to `upto` only (no look-ahead).

    Series with more than 10% missing in the window are dropped; remaining gaps are
    filled with 0 after standardizing (i.e. with the window mean).
    """
    w = panel.loc[TRAIN_FROM:upto]
    w = w.loc[:, w.isna().mean() <= 0.10]
    mu, sd = w.mean(), w.std().replace(0, np.nan)
    z = ((w - mu) / sd).dropna(axis=1, how="all").fillna(0.0)
    _, _, vt = np.linalg.svd(z.values, full_matrices=False)
    load = vt[:k].T
    zall = ((panel[z.columns] - mu[z.columns]) / sd[z.columns]).fillna(0.0)
    f = zall.values @ load
    return pd.DataFrame(f, index=panel.index, columns=[f"F{i+1}" for i in range(k)])


# ---------------------------------------------------------------- models
def gb():
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                     subsample=0.8, random_state=SEED)


def rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                 random_state=SEED, n_jobs=-1)


def fit_predict(model, X: pd.DataFrame, y: pd.Series, train_end: pd.Timestamp,
                h: int, origins: list[pd.Timestamp], base: pd.Series | None = None) -> pd.Series:
    """Direct h-step model: X at t predicts y at t+h. Training pairs need t+h <= train_end.

    If `base` is given, the model learns y(t+h) - base(t) and base(t) is added back,
    i.e. it predicts the change from today's 12-month inflation (the random-walk forecast).
    """
    yt = y.shift(-h) if base is None else y.shift(-h) - base
    tr = X.loc[TRAIN_FROM:train_end].join(yt.rename("y")).dropna()
    tr = tr[tr.index + pd.DateOffset(months=h) <= train_end]
    model.fit(tr.drop(columns="y"), tr["y"])
    Xo = X.loc[origins]
    ok = Xo.notna().all(axis=1)
    out = pd.Series(np.nan, index=origins)
    if ok.any():
        out[ok[ok].index] = model.predict(Xo[ok])
    if base is not None:
        out = out + base.reindex(origins).values
    return out


def run_horizon(h, pi12, panel, lags, ar, named):
    last = pi12.dropna().index[-1]
    origins_all = [t for t in pi12.loc[START:].index if t + pd.DateOffset(months=h) <= last]
    preds = {k: [] for k in ("AR", "FAAR", "GB", "GB+X", "GB+F", "RF+F")}
    if TARGET == "change":      # trees predict the change from today; also see pi12 itself
        base, L = pi12, lags.join(pi12.rename("pi12"))
    else:                       # trees predict the level of 12-month inflation
        base, L = None, lags
    for b in range(0, len(origins_all), REFIT_EVERY):
        block = origins_all[b:b + REFIT_EVERY]
        t0 = block[0]
        F = factors(panel, t0, N_FACTORS)
        preds["AR"].append(fit_predict(LinearRegression(), ar, pi12, t0, h, block))
        preds["FAAR"].append(fit_predict(LinearRegression(), ar.join(F), pi12, t0, h, block))
        preds["GB"].append(fit_predict(gb(), L, pi12, t0, h, block, base))
        preds["GB+X"].append(fit_predict(gb(), L.join(named), pi12, t0, h, block, base))
        preds["GB+F"].append(fit_predict(gb(), L.join(F), pi12, t0, h, block, base))
        preds["RF+F"].append(fit_predict(rf(), L.join(F), pi12, t0, h, block, base))
    origin = pd.DatetimeIndex(origins_all)
    target = origin + pd.DateOffset(months=h)
    res = pd.DataFrame({k: pd.concat(v).values for k, v in preds.items()}, index=target)
    res.index.name = "target_month"
    res.insert(0, "RW", pi12.loc[origin].values)
    res.insert(0, "actual", pi12.reindex(target).values)
    res["AVG"] = (res["AR"] + res["GB+F"]) / 2
    return res.dropna()


# ---------------------------------------------------------------- evaluation
def dm_test(e1: np.ndarray, e2: np.ndarray, h: int) -> float:
    """Diebold-Mariano p-value (two-sided), squared loss, Newey-West with h-1 lags."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for L in range(1, h):
        v += 2 * (1 - L / h) * (dc[L:] @ dc[:-L]) / n
    if v <= 0:
        return np.nan
    t = d.mean() / np.sqrt(v / n)
    return float(math.erfc(abs(t) / math.sqrt(2)))


def evaluate(results: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    models = [c for c in results[1].columns if c not in ("actual", "RW")]
    for h, f in results.items():
        for name, a, b in PERIODS:
            g = f.loc[a:b]
            erw = (g.RW - g.actual).values
            rmse_rw = np.sqrt(np.mean(erw ** 2))
            row = {"horizon": h, "period": name, "n": len(g), "RMSE_RW": round(rmse_rw, 3)}
            for m in models:
                em = (g[m] - g.actual).values
                row[m] = round(float(np.sqrt(np.mean(em ** 2)) / rmse_rw), 2)
                row[m + "_p"] = round(dm_test(em, erw, h), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df, tcodes = load_panel()
    pi12, panel, lags, ar, named = build(df, tcodes)
    missing = [c for c in NAMED_X if c not in df.columns]
    log = [f"Panel: {df.shape[1]} series, {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}",
           f"Last month with CPI: {pi12.dropna().index[-1]:%Y-%m}",
           f"Named drivers missing from this vintage: {missing or 'none'}"]
    print("\n".join(log))
    results = {}
    for h in HORIZONS:
        print(f"horizon {h} ...", flush=True)
        results[h] = run_horizon(h, pi12, panel, lags, ar, named)
        results[h].round(4).to_csv(OUT / f"forecasts_h{h}.csv")
    table = evaluate(results)
    table.to_csv(OUT / "rmse_table.csv", index=False)
    models = [c for c in results[1].columns if c not in ("actual", "RW")]
    print("\nRMSE relative to random walk (below 1 = better)")
    print(table[["horizon", "period", "n"] + models].to_string(index=False))
    (OUT / "run_log.txt").write_text("\n".join(log) + "\n\n" +
                                     table.to_string(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""How much of machine learning's advantage is the base effect?  (see DESIGN.md)

Reads the forecasts saved by forecast.py (change target) and ar12_check.py, adds the
RW-B benchmark, and reports for each model, horizon and period:
    rel_RW     RMSE / RMSE(RW)
    rel_RWB    RMSE / RMSE(RW-B)      gain beyond the base effect
    share_B    [RMSE(RW) - RMSE(RW-B)] / [RMSE(RW) - RMSE(M)], only when M beats RW
    p_RWB      Diebold-Mariano p-value against RW-B, Holm-corrected across models

    python forecast.py && python ar12_check.py && python base_effect.py
        -> results/change/base_effect.csv
"""
import numpy as np
import pandas as pd

import forecast as fc

MODELS = ["AR", "FAAR", "AR12", "GB", "GB+X", "GB+F", "RF+F", "AVG"]


def holm(p: dict) -> dict:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted((v, k) for k, v in p.items() if not np.isnan(v))
    m, out, run = len(items), {}, 0.0
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return {k: out.get(k, np.nan) for k in p}


def rw_b(pi12: pd.Series, pim: pd.Series, h: int) -> pd.Series:
    """RW-B forecast of pi12(t+h), indexed by origin t."""
    # base effect: monthly rates t-11 .. t+h-12 drop out of the 12-month window
    B = sum(pim.shift(12 - j) for j in range(1, h + 1))
    mbar = pim.rolling(12).mean()
    return pi12 - B + h * mbar


def main():
    df, tcodes = fc.load_panel()
    cpi = df["CPI"]
    pi12 = 100 * (cpi / cpi.shift(12) - 1)
    pim = 100 * (cpi / cpi.shift(1) - 1)
    ar12 = None
    p = fc.OUT / "ar12_forecasts.csv"
    if p.exists():
        ar12 = pd.read_csv(p, index_col=0, parse_dates=True)
    rows = []
    for h in fc.HORIZONS:
        f = pd.read_csv(fc.OUT / f"forecasts_h{h}.csv",
                        index_col=0, parse_dates=True)
        origin = f.index - pd.DateOffset(months=h)
        f["RW-B"] = rw_b(pi12, pim, h).reindex(origin).values
        if ar12 is not None and f"h{h}" in ar12:
            f["AR12"] = ar12[f"h{h}"].reindex(f.index)
        f = f.dropna()
        for name, a, b in fc.PERIODS:
            g = f.loc[a:b]
            e = {m: (g[m] - g.actual).values for m in ["RW", "RW-B"] + [m for m in MODELS if m in g]}
            r = {m: float(np.sqrt(np.mean(v ** 2))) for m, v in e.items()}
            praw = {m: fc.dm_test(e[m], e["RW-B"], h) for m in MODELS if m in e}
            padj = holm(praw)
            base_row = {"horizon": h, "period": name, "model": "RW-B",
                        "rel_RW": round(r["RW-B"] / r["RW"], 2), "rel_RWB": 1.0,
                        "share_B": np.nan, "p_RWB_holm": np.nan,
                        "p_RWB_vs_RW": round(fc.dm_test(e["RW-B"], e["RW"], h), 3)}
            rows.append(base_row)
            for m in MODELS:
                if m not in e:
                    continue
                gain = r["RW"] - r[m]
                share = (r["RW"] - r["RW-B"]) / gain if gain > 0 else np.nan
                rows.append({"horizon": h, "period": name, "model": m,
                             "rel_RW": round(r[m] / r["RW"], 2),
                             "rel_RWB": round(r[m] / r["RW-B"], 2),
                             "share_B": round(share, 2) if share == share else np.nan,
                             "p_RWB_holm": round(padj[m], 3), "p_RWB_vs_RW": np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(fc.OUT / "base_effect.csv", index=False)
    wide = t.pivot_table(index=["horizon", "period"], columns="model", values="rel_RWB", sort=False)
    print("RMSE relative to RW-B (the random walk that knows the base effect); below 1 = better")
    print(wide[["RW-B"] + [m for m in MODELS if m in wide]].to_string())
    rwb = t[t.model == "RW-B"][["horizon", "period", "rel_RW", "p_RWB_vs_RW"]]
    print("\nRW-B relative to RW")
    print(rwb.to_string(index=False))
    share = t.pivot_table(index=["horizon", "period"], columns="model", values="share_B", sort=False)
    print("\nBase-effect share of each model's gain over RW (1 = all of it)")
    print(share[[m for m in MODELS if m in share]].to_string())


if __name__ == "__main__":
    main()

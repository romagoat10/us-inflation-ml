"""Experiment 7: forecast monthly inflation and add the forecasts up (the "Medeiros way").

The main experiments forecast 12-month inflation directly, so the models have to learn the
base effect. Here every model forecasts the MONTHLY rate m(t+j), j = 1..12, with one direct
model per j (as in Medeiros et al., 2021). The forecast of 12-month inflation h months ahead
then combines the months that are already known with the forecast months:

    pi12(t+h) = 100 * [ prod over the 12 months t+h-11 .. t+h of (1 + m/100) - 1 ]
                known: m(t+h-11) .. m(t)          forecast: m(t+1) .. m(t+h)

so the base effect is built in for every model. The question is whether machine learning
then beats RW-B, the base-effect random walk.

Models (settings copied from forecast.py, no tuning):
    AR-m     linear regression of m(t+j) on the last 12 monthly rates and pi12(t)
    GB-m     gradient boosting on the same inputs
    GB-m+F   gradient boosting plus the 5 factors
    RF-m+F   random forest plus the 5 factors (closest to Medeiros et al.'s random forest)

Comparisons, for each horizon and period: RMSE relative to RW-B with Diebold-Mariano tests
(Holm-corrected across the four models), and against the direct models GB and AR12 from the
main run in the same results folder.

    set US_DATA=fredmd and US_FACTORS=em, then
    python forecast.py && python ar12_check.py && python monthly.py
        -> results/fredmd_monthly_em/
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import base_effect as be
import forecast as fc

MAX_J = max(fc.HORIZONS)
OUT = fc.ROOT / "results" / (("fredmd_monthly" if fc.DATASET == "fredmd" else "monthly")
                             + ("_em" if fc.FACTORS == "em" else ""))
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"AR-m": lambda: LinearRegression(), "GB-m": fc.gb, "GB-m+F": fc.gb, "RF-m+F": fc.rf}


def monthly_forecasts(pim, pi12, panel, lags, end=None):
    """Forecasts of m(t+j) for every origin t and j = 1..12. Returns {model: DataFrame[t, j]}."""
    last = pim.dropna().index[-1] if end is None else pd.Timestamp(end)
    origins = [t for t in pim.loc[fc.START:last].index if t < last]
    X = lags.join(pi12.rename("pi12"))
    out = {m: pd.DataFrame(np.nan, index=pd.DatetimeIndex(origins), columns=range(1, MAX_J + 1))
           for m in MODELS}
    for b in range(0, len(origins), fc.REFIT_EVERY):
        block = origins[b:b + fc.REFIT_EVERY]
        t0 = block[0]
        XF = X.join(fc.factors(panel, t0, fc.N_FACTORS))
        for j in range(1, MAX_J + 1):
            for m, make in MODELS.items():
                inputs = XF if m.endswith("+F") else X
                out[m].loc[block, j] = fc.fit_predict(make(), inputs, pim, t0, j, block).values
        if b % 30 == 0:
            print(f"  refit {t0:%Y-%m}", flush=True)
    return out


def combine(mhat: pd.DataFrame, pim: pd.Series, h: int) -> pd.Series:
    """12-month inflation forecast for t+h, indexed by target month."""
    g = 1 + pim / 100
    rows = {}
    for t in mhat.index:
        pos = pim.index.get_loc(t)
        known = g.iloc[pos - (11 - h): pos + 1] if h < 12 else g.iloc[0:0]
        fut = 1 + mhat.loc[t, list(range(1, h + 1))].astype(float) / 100
        val = 100 * (np.prod(known.values) * np.prod(fut.values) - 1)
        rows[t + pd.DateOffset(months=h)] = val
    return pd.Series(rows)


def main():
    df, tcodes = fc.load_panel()
    pi12, panel, lags, ar, named = fc.build(df, tcodes)
    pim = 100 * (df["CPI"] / df["CPI"].shift(1) - 1)
    print(f"Monthly-rate experiment: data {fc.DATASET}, factors {fc.FACTORS} -> {OUT}", flush=True)
    mh = monthly_forecasts(pim, pi12, panel, lags)
    for m, d in mh.items():
        d.round(5).to_csv(OUT / f"monthly_{m}.csv", index_label="origin")
    ar12 = pd.read_csv(fc.OUT / "ar12_forecasts.csv", index_col=0, parse_dates=True)
    rows = []
    for h in fc.HORIZONS:
        f = pd.read_csv(fc.OUT / f"forecasts_h{h}.csv", index_col=0, parse_dates=True)[["actual", "RW", "GB"]]
        f["AR12"] = ar12[f"h{h}"].reindex(f.index)
        f["RW-B"] = be.rw_b(pi12, pim, h).reindex(f.index - pd.DateOffset(months=h)).values
        for m, d in mh.items():
            f[m] = combine(d, pim, h).reindex(f.index)
        f = f.dropna()
        f.round(4).to_csv(OUT / f"forecasts_h{h}.csv")
        for name, a, b in fc.PERIODS:
            g = f.loc[a:b]
            e = {c: (g[c] - g.actual).values for c in g.columns if c != "actual"}
            r = {c: float(np.sqrt(np.mean(v ** 2))) for c, v in e.items()}
            p = be.holm({m: fc.dm_test(e[m], e["RW-B"], h) for m in MODELS})
            row = {"horizon": h, "period": name, "n": len(g)}
            for m in MODELS:
                row[f"{m}_vs_RWB"] = round(r[m] / r["RW-B"], 2)
                row[f"{m}_p_holm"] = round(p[m], 3)
            row["GB-m_vs_GB_direct"] = round(r["GB-m"] / r["GB"], 2)
            row["p_GB-m_vs_GB"] = round(fc.dm_test(e["GB-m"], e["GB"], h), 3)
            row["AR-m_vs_AR12"] = round(r["AR-m"] / r["AR12"], 2)
            row["p_AR-m_vs_AR12"] = round(fc.dm_test(e["AR-m"], e["AR12"], h), 3)
            row["RF-m+F_vs_AR-m"] = round(r["RF-m+F"] / r["AR-m"], 2)
            row["p_RF-m+F_vs_AR-m"] = round(fc.dm_test(e["RF-m+F"], e["AR-m"], h), 3)
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "monthly_results.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nRMSE relative to RW-B (below 1 = better), Holm-adjusted p in the _p_holm columns")
    print(t[["horizon", "period", "n"] + [c for c in t.columns if c.endswith("_vs_RWB") or c.endswith("_p_holm")]].to_string(index=False))
    print("\nMonthly-rate models against the direct models of the main run")
    print(t[["horizon", "period", "GB-m_vs_GB_direct", "p_GB-m_vs_GB", "AR-m_vs_AR12", "p_AR-m_vs_AR12",
             "RF-m+F_vs_AR-m", "p_RF-m+F_vs_AR-m"]].to_string(index=False))
    (OUT / "run_log.txt").write_text(t.to_string(index=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

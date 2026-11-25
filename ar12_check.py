"""Fairness check: a linear AR that sees the same 12 monthly lags as the trees.

At short horizons, 12-month inflation next month = today's 12-month rate + (next month's
monthly rate - the monthly rate 11 months ago). A model that sees lag 11 knows the
"base effect" that drops out. The trees see all 12 lags; the AR in forecast.py sees only 3.
This script fits AR12 (OLS on the 12 monthly lags and 12-month inflation, predicting the
change from today) with the same expanding window and refit schedule, and compares it with
RW and with the trees.

    python forecast.py && python ar12_check.py    -> results/change/ar12_check.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import forecast as fc

fc.TARGET = "change"
df, tcodes = fc.load_panel()
pi12, panel, lags, ar, named = fc.build(df, tcodes)
X = lags.join(pi12.rename("pi12"))
rows, saved = [], {}
for h in fc.HORIZONS:
    f = pd.read_csv(fc.OUT / f"forecasts_h{h}.csv", index_col=0, parse_dates=True)
    last = pi12.dropna().index[-1]
    origins_all = [t for t in pi12.loc[fc.START:].index if t + pd.DateOffset(months=h) <= last]
    parts = []
    for b in range(0, len(origins_all), fc.REFIT_EVERY):
        block = origins_all[b:b + fc.REFIT_EVERY]
        parts.append(fc.fit_predict(LinearRegression(), X, pi12, block[0], h, block, pi12))
    p = pd.concat(parts)
    p.index = pd.DatetimeIndex(origins_all) + pd.DateOffset(months=h)
    saved[f"h{h}"] = p
    f["AR12"] = p.reindex(f.index)
    f = f.dropna()
    for name, a, b in fc.PERIODS:
        g = f.loc[a:b]
        e = {m: (g[m] - g.actual).values for m in ("RW", "AR12", "GB", "GB+F")}
        r = lambda m: np.sqrt(np.mean(e[m] ** 2))
        rows.append({"horizon": h, "period": name,
                     "AR12_vs_RW": round(r("AR12") / r("RW"), 2),
                     "p_AR12_vs_RW": round(fc.dm_test(e["AR12"], e["RW"], h), 3),
                     "GB_vs_AR12": round(r("GB") / r("AR12"), 2),
                     "p_GB_vs_AR12": round(fc.dm_test(e["GB"], e["AR12"], h), 3),
                     "GBF_vs_AR12": round(r("GB+F") / r("AR12"), 2),
                     "p_GBF_vs_AR12": round(fc.dm_test(e["GB+F"], e["AR12"], h), 3)})
pd.DataFrame(saved).to_csv(fc.OUT / "ar12_forecasts.csv", index_label="target_month")
t = pd.DataFrame(rows)
t.to_csv(fc.OUT / "ar12_check.csv", index=False)
print(t.to_string(index=False))

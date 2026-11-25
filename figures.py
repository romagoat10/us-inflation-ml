"""Figures for the paper.

    figure_forecasts_h{h}.png   actual 12-month inflation and the RW, RW-B and GB forecasts
    figure_rolling.png          rolling 36-month RMSE of GB and AR12 relative to RW-B, per horizon

Uses the results folder of the current dataset (set US_DATA=fredmd for FRED-MD).

    python forecast.py && python ar12_check.py && python figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import base_effect as be
import forecast as fc

WINDOW = 36  # months


def load(h: int) -> pd.DataFrame:
    df, _ = fc.load_panel()
    cpi = df["CPI"]
    pi12 = 100 * (cpi / cpi.shift(12) - 1)
    pim = 100 * (cpi / cpi.shift(1) - 1)
    f = pd.read_csv(fc.OUT / f"forecasts_h{h}.csv", index_col=0, parse_dates=True)
    f["RW-B"] = be.rw_b(pi12, pim, h).reindex(f.index - pd.DateOffset(months=h)).values
    ar12 = fc.OUT / "ar12_forecasts.csv"
    if ar12.exists():
        f["AR12"] = pd.read_csv(ar12, index_col=0, parse_dates=True)[f"h{h}"].reindex(f.index)
    return f.dropna()


def forecasts_figure(h: int):
    f = load(h).loc["2015":]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(f.index, f.actual, color="black", lw=2, label="Actual")
    ax.plot(f.index, f.RW, color="#999999", lw=1.2, ls="--", label="Random walk")
    ax.plot(f.index, f["RW-B"], color="#1f77b4", lw=1.2, label="Random walk + base effect")
    ax.plot(f.index, f.GB, color="#d62728", lw=1.2, label="Gradient boosting")
    ax.set_ylabel("12-month CPI inflation, %")
    ax.set_title(f"US inflation and forecasts made {h} months earlier")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fc.OUT / f"figure_forecasts_h{h}.png", dpi=160)
    plt.close(fig)


def rolling_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, h in zip(axes.ravel(), fc.HORIZONS):
        f = load(h)
        base = ((f["RW-B"] - f.actual) ** 2).rolling(WINDOW).mean()
        for m, c in (("GB", "#d62728"), ("AR12", "#2ca02c")):
            if m in f:
                rel = np.sqrt(((f[m] - f.actual) ** 2).rolling(WINDOW).mean() / base)
                ax.plot(rel.index, rel, color=c, lw=1.3, label=m)
        ax.axhline(1, color="black", lw=0.8)
        ax.set_title(f"{h} month{'s' if h > 1 else ''} ahead")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("RMSE relative to RW-B")
    axes[1, 0].set_ylabel("RMSE relative to RW-B")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Rolling {WINDOW}-month accuracy relative to the base-effect random walk (below 1 = better)")
    fig.tight_layout()
    fig.savefig(fc.OUT / "figure_rolling.png", dpi=160)
    plt.close(fig)


def main():
    for h in (3, 12):
        forecasts_figure(h)
    rolling_figure()
    print("saved figures to", fc.OUT)


if __name__ == "__main__":
    main()

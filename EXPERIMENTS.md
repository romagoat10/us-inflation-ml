# Experiment log

Every experiment in this project, in the order it was run. Each entry gives why it was run, what changed from the one before, what it found, and what it led to. Nothing was deleted: the results of every run are kept in `results/`.

| # | Experiment | Data | Results folder |
|---|---|---|---|
| 0 | Kazakhstan pilot | IMF CPI, 2010–2025 | separate repo: [-kazakhstan-inflation-ml](https://github.com/romagoat10/-kazakhstan-inflation-ml) |
| 1 | Trees predict the level | 21 US series (DBnomics) | `results/levels/` |
| 2 | Trees predict the change | 21 US series | `results/change/` |
| 3 | Fairness check with a linear AR12 | 21 US series | `results/change/ar12_check.csv` |
| 4 | Base-effect benchmark (RW-B) | 21 US series | `results/change/base_effect.csv` |
| 5 | **Confirmatory run** | full FRED-MD (126 series) | `results/fredmd_change/` |
| 6 | Robustness: FRED-MD factor step | full FRED-MD | `results/fredmd_change_em/` |
| 7 | Follow-up: forecast monthly rates | full FRED-MD | `results/fredmd_monthly_em/` |

Experiments 1–4 are **exploratory**: they shaped the design. Experiment 5 is **confirmatory**: the design was written down in [DESIGN.md](DESIGN.md) before it was run, and it was run once. Experiments 6 and 7 were added afterwards, and they are labeled as such.

---

## 0. Kazakhstan pilot

- **Why:** a first test of the idea that machine learning can forecast inflation better than traditional models.
- **What:** random walk, AR, gradient boosting, and gradient boosting with the exchange rate and oil prices, for Kazakhstan's CPI, forecasts from 2010.
- **Found:** gradient boosting and AR beat the random walk by about 10–25%, but only 1 of 24 comparisons was significant. The exchange rate and oil did not help.
- **Led to:** too little data to say much. Move to the US, where there are more than 25 years of forecasts to test.

## 1. Trees predict the level of inflation

- **Why:** the first US run, with the simplest setup.
- **What:** 21 US series from DBnomics; the tree models predict the level of 12-month inflation h months ahead.
- **Found:** the trees were **2–3 times worse** than the random walk at 1 month (gradient boosting 2.31 in 2008–19). Two reasons: trees cannot add up 12 monthly rates, and they cannot predict values outside the training range, such as the 2009 deflation.
- **Led to:** make the trees predict the change from today's inflation instead, which is standard in the forecasting literature. This run is kept, so the switch is visible.

## 2. Trees predict the change

- **Why:** to fix the problem found in experiment 1.
- **What:** same data and models; the trees now predict the change π(t+h) − π(t), which is added back to today's rate.
- **Found:** every model that uses the last 12 monthly rates beat the random walk by **20–45% at 1–3 months**, significantly at 1 month.
- **Led to:** the gains looked too good. A fair linear model with the same inputs was needed before crediting machine learning.

## 3. Fairness check: linear AR12

- **Why:** to check whether the gain comes from machine learning or only from the inputs.
- **What:** a linear regression on today's inflation and the last 12 monthly rates (AR12), with the same inputs as the trees.
- **Found:** **AR12 was as accurate as the trees at 1–6 months.** Machine learning was never significantly better than AR12 there. The one exception was 12 months ahead in 2000–07, where GB+F was 27% better than AR12 (p < 0.001).
- **Led to:** the hypothesis that the short-horizon gain is the **base effect**: part of the coming change in 12-month inflation is already known, because old monthly rates drop out of the 12-month window. Any model that sees those rates gets this for free.

## 4. Base-effect benchmark (RW-B)

- **Why:** to measure the base effect directly.
- **What:** a new benchmark, RW-B: the random walk minus the base effect, plus the recent average monthly rate. No estimation at all. At 12 months it equals the random walk.
- **Found:** RW-B alone was **20–35% better than the random walk** at 1–3 months. Before 2020, no model beat RW-B at 1–6 months. After 2020, gradient boosting beat it by 12–14% at 3 and 6 months, but AR12 did equally well.
- **Led to:** a clear question that could be tested properly. It was written down in [DESIGN.md](DESIGN.md) (question, benchmarks, measures, settings) before the confirmatory run.

## 5. Confirmatory run on FRED-MD

- **Why:** experiments 1–4 had shaped the design, so they could not confirm it. A new, larger dataset and a fixed design were needed.
- **What:** the same code and settings on the full FRED-MD panel (126 series, October 2026 vintage), run once.
- **Found:**
  - RW-B is 20–35% better than the random walk at 1–3 months.
  - **No machine learning model is significantly better than RW-B** in any horizon and period after the Holm correction, except gradient boosting at 3 months after 2020 (0.86, p = 0.006). AR12 is equally accurate there (0.86).
  - **The 12-month machine learning win from experiment 3 did not replicate** (0.96, p = 0.61).
  - The 126 series did not help: the factor-augmented models were often worse.
- **Led to:** the main result of the paper. One open concern: the factor step was simple (mean-filled gaps, no outlier treatment), so the weak factor results might be the method's fault.

## 6. Robustness: FRED-MD factor step

- **Why:** to answer the concern from experiment 5.
- **What:** the factor treatment recommended by McCracken and Ng (2016): outliers more than 10 interquartile ranges from the median are removed, gaps are filled with the EM algorithm, and the most recent months are handled by least squares. Everything else is unchanged.
- **Found:** 162 outliers removed by 2025, 80 of them in 2020–21. After 2020, the factor model's error fell by 16–23% and GB+F's by up to 12%. Before 2020 nothing changed by more than 0.04. **The main conclusions did not change.**
- **Led to:** the weak factor results are not caused by the outlier and missing-value treatment.

---

## 7. Follow-up: forecast monthly inflation and add it up

- **Why:** the direct models have to learn the base effect, which may handicap the trees. Medeiros et al. (2021) forecast monthly inflation instead, so this is a fairer test of machine learning and a direct link to their paper.
- **What:** each model forecasts the monthly rate 1 to 12 months ahead (one direct model per month), with the same inputs, settings and FRED-MD factor step as experiment 6. The forecasts are combined with the months already known into 12-month inflation, so every model has the base effect built in. Check: with perfect monthly forecasts the combination reproduces 12-month inflation exactly. Script: `monthly.py`.
- **Found:**
  - The tree models beat RW-B significantly in 8 of 48 cells: at 1, 3 and 6 months in 2000–07 (gains of 10–17%) and at 3 months after 2020 (12–13%). None is significantly worse.
  - The linear model in the same setup (AR-m) never beats RW-B significantly and is within 3% of AR12, so the gain is specific to machine learning.
  - Forecasting monthly rates makes gradient boosting 10–12% more accurate than the direct version before 2020.
  - Nothing is significant in 2008–19 or at 12 months.
- **Led to:** a refined conclusion. Most of machine learning's edge over the random walk is the base effect, but a real, modest edge of 10–17% remains in some periods when the trees forecast monthly rates. This was found after the main run, so it needs confirming on new data.

---

## Next experiments (planned, not run yet)

1. Real-time FRED-MD vintages instead of revised data. This would also confirm experiment 7 on new data.
2. LASSO, ridge and a small neural network, tuned on a hold-out period before 2000.
3. Core CPI and PCE inflation as targets.
4. The same test for the UK and Kazakhstan.

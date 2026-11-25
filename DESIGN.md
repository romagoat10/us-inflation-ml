# Research design (written 25 November 2026, before the confirmatory run)

## Question

**How much of machine learning's advantage in forecasting 12-month inflation is the "base effect"?**

12-month inflation is a sum of the last 12 monthly rates. When the horizon is shorter than 12 months, part of the future change is already known today:

    pi12(t+h) - pi12(t)  ≈  [m(t+1) + ... + m(t+h)]  -  [m(t-11) + ... + m(t+h-12)]
                              unknown: future months      known today: months that drop out

`m` is monthly inflation. The second bracket is the **base effect** B(t,h). Any model that sees the last 12 monthly rates can use it, including a linear regression. The first run of this project suggested that most of the advantage of machine learning over the random walk at 1–6 months comes from B, not from learning anything about future months.

## Benchmarks

| Name | Forecast of pi12(t+h) | Knows the base effect? |
|---|---|---|
| RW | pi12(t) | no |
| RW-B | pi12(t) − B(t,h) + h·m̄(t), where m̄(t) is the average monthly rate over the last 12 months | yes, and nothing else |
| AR12 | linear regression on pi12(t) and the last 12 monthly rates (predicts the change) | yes, plus linear dynamics |

At h = 12, RW-B equals RW: there is no base effect left to exploit. That gives a built-in prediction: **if the advantage of machine learning is mostly the base effect, its gain over RW-B should be small at 1–6 months, and its real skill should show up only at 12 months.**

## Measures, fixed in advance

For each model M, horizon h and period:

1. RMSE(M) / RMSE(RW), the usual headline number.
2. RMSE(M) / RMSE(RW-B), the gain beyond the base effect.
3. **Base-effect share** of the gain over RW = [RMSE(RW) − RMSE(RW-B)] / [RMSE(RW) − RMSE(M)]. It is reported only when M beats RW. A value of 1 means the whole gain is the base effect.
4. Diebold–Mariano tests (Newey–West variance, h−1 lags) of M against RW, RW-B and AR12, with a **Holm correction** across the models within each horizon and period.

## Fixed choices

- Horizons 1, 3, 6, 12. Periods 2000–07, 2008–19, 2020–end.
- Training from 1973 (21-series panel) or 1962 (FRED-MD); expanding window; refit every 3 months.
- The models and settings already in `forecast.py`. No tuning.
- Tree models predict the change from pi12(t), with pi12(t) as an input.
- 5 factors, from principal components on data available at each date.

## Which run counts

- The **exploratory** runs are the 21-series DBnomics panel (`results/levels`, `results/change`, `ar12_check.py`). They shaped this design.
- The **confirmatory** run is the same code on the **full FRED-MD panel**, run once with the choices above. Whatever it shows will be reported.
- Later: real-time FRED-MD vintages, and the same analysis for Kazakhstan.

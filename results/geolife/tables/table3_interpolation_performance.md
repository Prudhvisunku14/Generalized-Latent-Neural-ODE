| Method | RMSE median [95% CI] (m) | RMSE mean (m) | MAE median (m) | ADE median (m) | median err. (m) | p90 err. (m) | max err. (m) | win rate vs linear | time / segment (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Latent ODE (frozen, three-shape) | 126.40 [110.94, 147.94] | 272.70 | 93.29 | 148.75 | 121.60 | 286.59 | 443.72 | 0.1% | 26.26 |
| Linear | 7.10 [5.80, 8.48] | 12.99 | 3.88 | 6.33 | 3.14 | 15.52 | 32.36 | — | 0.02 |
| Cubic spline | 5.56 [4.64, 6.90] | 11.65 | 3.05 | 4.92 | 2.45 | 12.27 | 28.18 | 68.6% | 0.20 |
| PCHIP | 5.62 [4.54, 6.69] | 10.11 | 3.09 | 4.97 | 2.54 | 12.41 | 26.70 | 85.9% | 0.21 |

| Paired comparison (per-segment RMSE) | geometric-mean ratio [95% user-bootstrap CI] | median ratio | win rate (first lower) | Wilcoxon p (Holm) |
|---|---|---|---|---|
| latent_ode vs linear | 17.278 [15.388, 19.345] | 16.396 | 0.1% | 1.3e-164 |
| latent_ode vs cubic | 21.619 [18.594, 24.967] | 21.402 | 0.8% | 5.7e-163 |
| latent_ode vs pchip | 22.221 [19.414, 25.232] | 20.771 | 0.0% | 1.7e-164 |
| cubic vs linear | 0.799 [0.768, 0.835] | 0.829 | 68.6% | 9.8e-39 |
| pchip vs linear | 0.778 [0.758, 0.797] | 0.812 | 85.9% | 3.2e-116 |

| Forecast (secondary) | protocol | targets | median horizon (s) | Latent ODE RMSE median (m) | hold-last (m) | constant velocity (m) | LODE win rate vs const. vel. |
|---|---|---|---|---|---|---|---|
| | A | 61,485 | 142 | 702.2 | 228.9 | 151.4 | 8.2% |
| | B | 57,556 | 124 | 661.4 | 198.6 | 121.4 | 7.5% |

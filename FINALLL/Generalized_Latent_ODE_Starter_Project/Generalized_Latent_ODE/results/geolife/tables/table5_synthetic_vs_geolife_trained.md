| Model (30 observations, held-out test users) | Interp. RMSE median [95% CI] (m) | mean (m) | win rate vs linear | win rate vs PCHIP | geo-mean ratio vs linear | Forecast A median (m) | Forecast B median (m) |
|---|---|---|---|---|---|---|---|
| Linear | 7.10 [5.80, 8.48] | 12.99 | 0.0% | 14.1% | 1.000 | — | — |
| Cubic spline | 5.56 [4.64, 6.90] | 11.65 | 68.6% | 48.5% | 0.799 | — | — |
| PCHIP | 5.62 [4.54, 6.69] | 10.11 | 85.9% | 0.0% | 0.778 | — | — |
| Frozen synthetic (S3) | 126.40 [110.94, 147.94] | 272.70 | 0.1% | 0.0% | 17.278 | 702.2 | 661.4 |
| Frozen synthetic (S1) | 140.20 [119.71, 164.80] | 316.88 | 0.0% | 0.0% | 18.946 | 678.7 | 654.7 |
| GeoLife-trained (scratch) | 60.62 [51.58, 71.66] | 116.18 | 0.1% | 0.1% | 8.222 | 322.3 | 287.0 |
| Synthetic → GeoLife fine-tuned | 47.11 [42.24, 54.63] | 84.96 | 0.1% | 0.1% | 6.415 | 370.6 | 324.2 |
| Hold last (forecast baseline) | — | — | — | — | — | 228.9 | 198.6 |
| Constant velocity (forecast baseline) | — | — | — | — | — | 151.4 | 121.4 |

| Paired (30 obs) | geometric-mean RMSE ratio [95% user-bootstrap CI] | win rate | Wilcoxon p |
|---|---|---|---|
| geolife_scratch_vs_linear | 8.222 [7.434, 9.122] | 0.1% | 6.5e-165 |
| geolife_scratch_vs_pchip | 10.573 [9.374, 11.916] | 0.1% | 5.3e-165 |
| geolife_scratch_vs_frozen_three_shapes | 0.476 [0.456, 0.497] | 88.3% | 3e-128 |
| geolife_finetuned_from_three_shapes_vs_linear | 6.415 [5.889, 7.008] | 0.1% | 6.6e-165 |
| geolife_finetuned_from_three_shapes_vs_pchip | 8.250 [7.436, 9.154] | 0.1% | 5e-165 |
| geolife_finetuned_from_three_shapes_vs_frozen_three_shapes | 0.371 [0.354, 0.389] | 96.6% | 1.6e-160 |

| N obs | Linear | Cubic spline | PCHIP | Frozen synthetic (S3) | Frozen synthetic (S1) | GeoLife-trained (scratch) | Synthetic → GeoLife fine-tuned |
|---|---|---|---|---|---|---|---|
| 10 | 20.24 | 20.79 | 17.48 | 190.30 | 290.10 | 119.58 | 191.67 |
| 20 | 10.80 | 9.04 | 8.67 | 129.84 | 166.01 | 75.12 | 94.78 |
| 30 | 7.10 | 5.56 | 5.62 | 126.40 | 140.20 | 60.62 | 47.11 |
| 40 | 4.83 | 3.61 | 3.57 | 146.56 | 170.64 | 59.80 | 76.73 |
| 50 | 3.79 | 2.86 | 2.93 | 176.92 | 202.18 | 59.99 | 115.54 |

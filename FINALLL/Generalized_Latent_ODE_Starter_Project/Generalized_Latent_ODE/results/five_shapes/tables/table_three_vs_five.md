| Family | three-shape model (frozen) | five-shape model | Forecast RMSE: three-shape | Forecast RMSE: five-shape | difference [95% CI] | five-shape better on | Holm p | Recon. RMSE: three-shape | Recon. RMSE: five-shape | Holm p (recon.) |
|---|---|---|---|---|---|---|---|---|---|---|
| circle | trained on | trained on | 0.1401 | 0.2594 | +0.1193 [+0.0982, +0.1404] | 15% | 3e-20 | 0.0622 | 0.0701 | 0.00026 |
| spiral | trained on | trained on | 0.2908 | 0.4270 | +0.1362 [+0.0998, +0.1726] | 27% | 1.7e-11 | 0.0933 | 0.1131 | 4.3e-10 |
| ellipse | trained on | trained on | 0.1885 | 0.2609 | +0.0723 [+0.0536, +0.0911] | 21% | 1.7e-11 | 0.0888 | 0.0973 | 0.0048 |
| figure_eight | **unseen** | trained on | 1.1174 | 0.2992 | -0.8182 [-0.8940, -0.7424] | 99% | 2.3e-25 | 0.7321 | 0.1086 | 2.3e-25 |
| sinusoid | **unseen** | trained on | 0.9734 | 0.3861 | -0.5874 [-0.6212, -0.5536] | 99% | 2.3e-25 | 0.3441 | 0.0647 | 2.3e-25 |
| overall (750) | | | 0.5421 | 0.3265 | | | | 0.2641 | 0.0908 | |

| Family | five-shape recon. RMSE mean (median) | linear interp. | five-shape forecast RMSE mean (median) | hold-last | const. velocity |
|---|---|---|---|---|---|
| circle | 0.0701 (0.0643) | 0.0699 | 0.2594 (0.2342) | 1.1370 | 2.3274 |
| spiral | 0.1131 (0.1057) | 0.0653 | 0.4270 (0.3744) | 1.2763 | 2.3877 |
| ellipse | 0.0973 (0.0870) | 0.0637 | 0.2609 (0.2390) | 1.1200 | 2.0143 |
| figure_eight | 0.1086 (0.0970) | 0.0917 | 0.2992 (0.2589) | 0.9610 | 2.6176 |
| sinusoid | 0.0647 (0.0600) | 0.0452 | 0.3861 (0.3676) | 0.6248 | 1.3898 |
| overall | 0.0908 (0.0809) | 0.0672 | 0.3265 (0.2723) | 1.0238 | 2.1474 |

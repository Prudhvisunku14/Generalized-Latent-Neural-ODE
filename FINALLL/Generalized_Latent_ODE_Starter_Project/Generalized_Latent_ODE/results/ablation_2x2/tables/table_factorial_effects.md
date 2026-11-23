| Effect on per-trajectory RMSE | Group | Metric | mean [95% bootstrap CI] | Wilcoxon p (Holm) |
|---|---|---|---|---:|
| latent_main_effect (16 vs 8, averaged over GRU) | circle | recon | -0.0167 [-0.0194, -0.0142] | 7.1e-22 |
| gru_main_effect (128 vs 64, averaged over latent) | circle | recon | -0.0070 [-0.0091, -0.0050] | 1.6e-09 |
| interaction ((D-C) - (B-A)) | circle | recon | +0.0047 [+0.0010, +0.0083] | 0.046 |
| latent_main_effect (16 vs 8, averaged over GRU) | circle | forecast | +0.1703 [+0.1411, +0.1994] | 6.8e-16 |
| gru_main_effect (128 vs 64, averaged over latent) | circle | forecast | +0.0803 [+0.0540, +0.1060] | 8e-07 |
| interaction ((D-C) - (B-A)) | circle | forecast | +0.1193 [+0.0871, +0.1518] | 6.5e-09 |
| latent_main_effect (16 vs 8, averaged over GRU) | spiral | recon | -0.0347 [-0.0388, -0.0308] | 4.1e-24 |
| gru_main_effect (128 vs 64, averaged over latent) | spiral | recon | -0.0102 [-0.0131, -0.0074] | 2.3e-10 |
| interaction ((D-C) - (B-A)) | spiral | recon | +0.0141 [+0.0083, +0.0204] | 3.5e-06 |
| latent_main_effect (16 vs 8, averaged over GRU) | spiral | forecast | +0.1028 [+0.0553, +0.1517] | 0.1 |
| gru_main_effect (128 vs 64, averaged over latent) | spiral | forecast | +0.0662 [+0.0330, +0.1002] | 0.005 |
| interaction ((D-C) - (B-A)) | spiral | forecast | +0.1181 [+0.0696, +0.1700] | 0.00078 |
| latent_main_effect (16 vs 8, averaged over GRU) | ellipse | recon | -0.0385 [-0.0419, -0.0352] | 7.4e-25 |
| gru_main_effect (128 vs 64, averaged over latent) | ellipse | recon | -0.0153 [-0.0183, -0.0123] | 2.8e-15 |
| interaction ((D-C) - (B-A)) | ellipse | recon | +0.0250 [+0.0194, +0.0308] | 3.2e-13 |
| latent_main_effect (16 vs 8, averaged over GRU) | ellipse | forecast | +0.0881 [+0.0564, +0.1192] | 8.6e-07 |
| gru_main_effect (128 vs 64, averaged over latent) | ellipse | forecast | +0.0444 [+0.0183, +0.0704] | 0.005 |
| interaction ((D-C) - (B-A)) | ellipse | forecast | +0.2094 [+0.1727, +0.2474] | 1.3e-16 |
| latent_main_effect (16 vs 8, averaged over GRU) | figure_eight | recon | -0.0448 [-0.0500, -0.0395] | 1.2e-23 |
| gru_main_effect (128 vs 64, averaged over latent) | figure_eight | recon | -0.0349 [-0.0395, -0.0303] | 1.3e-22 |
| interaction ((D-C) - (B-A)) | figure_eight | recon | +0.0524 [+0.0441, +0.0609] | 2.3e-20 |
| latent_main_effect (16 vs 8, averaged over GRU) | figure_eight | forecast | +0.0153 [-0.0153, +0.0454] | 0.55 |
| gru_main_effect (128 vs 64, averaged over latent) | figure_eight | forecast | -0.0048 [-0.0398, +0.0312] | 0.65 |
| interaction ((D-C) - (B-A)) | figure_eight | forecast | +0.0764 [+0.0079, +0.1431] | 0.1 |
| latent_main_effect (16 vs 8, averaged over GRU) | sinusoid | recon | -0.0220 [-0.0245, -0.0196] | 8.8e-25 |
| gru_main_effect (128 vs 64, averaged over latent) | sinusoid | recon | -0.0105 [-0.0126, -0.0086] | 4.5e-17 |
| interaction ((D-C) - (B-A)) | sinusoid | recon | +0.0103 [+0.0068, +0.0141] | 9e-06 |
| latent_main_effect (16 vs 8, averaged over GRU) | sinusoid | forecast | +0.3458 [+0.3297, +0.3626] | 7.4e-25 |
| gru_main_effect (128 vs 64, averaged over latent) | sinusoid | forecast | +0.0606 [+0.0423, +0.0784] | 3e-08 |
| interaction ((D-C) - (B-A)) | sinusoid | forecast | -0.2801 [-0.3065, -0.2525] | 2.5e-23 |
| latent_main_effect (16 vs 8, averaged over GRU) | overall | recon | -0.0313 [-0.0332, -0.0296] | 1.1e-118 |
| gru_main_effect (128 vs 64, averaged over latent) | overall | recon | -0.0156 [-0.0172, -0.0140] | 3.3e-75 |
| interaction ((D-C) - (B-A)) | overall | recon | +0.0213 [+0.0185, +0.0242] | 8.2e-45 |
| latent_main_effect (16 vs 8, averaged over GRU) | overall | forecast | +0.1445 [+0.1273, +0.1612] | 6.6e-47 |
| gru_main_effect (128 vs 64, averaged over latent) | overall | forecast | +0.0494 [+0.0363, +0.0622] | 2.3e-11 |
| interaction ((D-C) - (B-A)) | overall | forecast | +0.0486 [+0.0251, +0.0726] | 0.0018 |

Positive = higher error. Intervals and tests resample test trajectories only (single training run per cell); they do not capture training-seed variability.

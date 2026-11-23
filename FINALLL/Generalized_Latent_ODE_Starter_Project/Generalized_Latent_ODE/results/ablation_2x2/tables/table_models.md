| Model | latent | GRU | parameters | seed / data seed | optimiser steps | selected epoch | val. -ELBO | val. forecast RMSE (pooled) | val. context RMSE (pooled) | train NLL | val NLL | min val. forecast RMSE (epoch) | training min |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| A  latent 8 / GRU 64 | 8 | 64 | 24,602 | 42 / 42 | 33,000 | 294 | 46.80 | 0.4989 | 0.0956 | 8.55 | 16.93 | 0.4976 (270) | 80.8 |
| B  latent 16 / GRU 64 | 16 | 64 | 27,186 | 42 / 42 | 33,000 | 299 | -22.97 | 0.5909 | 0.0454 | -63.83 | -69.32 | 0.5758 (224) | 80.6 |
| C  latent 8 / GRU 128 | 8 | 128 | 63,642 | 42 / 42 | 33,000 | 299 | -5.47 | 0.4950 | 0.0655 | -42.74 | -39.32 | 0.4446 (190) | 81.5 |
| D  latent 16 / GRU 128 | 16 | 128 | 67,250 | 42 / 42 | 33,000 | 296 | -26.55 | 0.6487 | 0.0404 | -67.03 | -72.27 | 0.6265 (152) | 81.5 |
| earlier baseline (8/64) | 8 | 64 | 24,602 | 42 / None | 33,000 | 297 | 48.50 | 0.3792 | 0.0942 | 2.33 | 14.28 | 0.3628 (203) | 77.9 |
| earlier medium (16/128) | 16 | 128 | 67,250 | 42 / None | 33,000 | 294 | -31.47 | 0.5531 | 0.0388 | -68.28 | -73.96 | 0.5504 (281) | 80.0 |
| earlier large (32/128) | 32 | 128 | 74,466 | 42 / None | 33,000 | 296 | -19.08 | 0.7035 | 0.0421 | -64.24 | -70.92 | 0.6149 (31) | 84.1 |

Selection criterion for every row: validation -ELBO only. The "min val. forecast RMSE" column is a post-hoc diagnostic on the validation split; no checkpoint was selected with it and intermediate checkpoints were not saved, so it cannot be evaluated on the test set. Train NLL is measured with a sampled z0 and stored noisy targets; val NLL uses the posterior mean, so the two are not strictly like-for-like.

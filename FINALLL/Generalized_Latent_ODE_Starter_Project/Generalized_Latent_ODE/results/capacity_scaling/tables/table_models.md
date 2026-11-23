| Model | latent | GRU | ODE / decoder hidden | trainable parameters | trained on | best epoch | best val. −ELBO | epochs run | training time (min) |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|
| Three-shape (frozen) | 8 | 64 | 64 / 64 | 24,602 | three_shapes | 298 | 20.12 | 300 | 59.6 |
| Baseline five-shape | 8 | 64 | 64 / 64 | 24,602 | five_shapes | 297 | 48.50 | 300 | 77.9 |
| Medium five-shape | 16 | 128 | 64 / 64 | 67,250 | five_shapes | 294 | -31.47 | 300 | 80.0 |
| Large five-shape | 32 | 128 | 64 / 64 | 74,466 | five_shapes | 296 | -19.08 | 300 | 84.1 |

Validation −ELBO is comparable only between the three five-shape models (same validation data); the three-shape model was validated on three-shape data.

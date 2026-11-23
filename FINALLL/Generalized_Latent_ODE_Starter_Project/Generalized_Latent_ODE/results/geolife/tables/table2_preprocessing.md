| Setting | Value |
|---|---|
| Projection | spherical azimuthal equidistant at the segment's first record (always observed), radius = WGS-84 Gaussian radius at that latitude |
| Run breaks | gap > 60 s or step speed > 70 m/s |
| Segment | 200 consecutive cleaned records, non-overlapping, extent ≥ 100 m, ≥ 60 context and ≥ 10 future records |
| Time (protocol A, primary) | t = 10 (t_real − t_start) / (t_end − t_start); context t ≤ 7, forecast 7 < t ≤ 10; irregular spacing kept |
| Time (protocol B, forecasting) | t = 7 (t_real − t_start) / (t_last_obs − t_start); no knowledge of the segment end |
| Observations | 30: the first record + 29 uniformly random other context records; seed = 100000 + 1000003·N + segment index |
| Interpolation targets | all withheld context records between the first and last observation (recorded GPS positions) |
| Coordinate transform | S3_obsmean_rms_p95: centre = obs_mean, spread = obs_rms, mapped to 1.665 model units; fitted on the observations only |
| Strategy selection | lowest MEDIAN per-segment interpolation RMSE (m) of the frozen Latent ODE on development users (declared before running); 300 dev segments |
| Splits | users split 70/30 test/dev (seed 20261115); ≤ 25 segments per user |
| Model | three_shapes_unified (frozen, posterior mean z0, RK4 step 0.1) via glo.inference.predict |

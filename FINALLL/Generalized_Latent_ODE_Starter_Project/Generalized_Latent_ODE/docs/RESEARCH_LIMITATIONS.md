# Research limitations

What the evidence in this repository does and does not support.


## Research limitations

1. **Single seed.** The five-family, capacity, 2x2, GeoLife and zero-shot results come from one training run per model. Run-to-run variation can be large: identical architectures differ by +22.4% and +20.7% in mean forecast RMSE between runs, and the five-seed two-family study has a forecast CV of about 23% (range 0.169 to 0.283). Test-trajectory p-values do **not** capture this.
2. **Synthetic data only (apart from GeoLife).** Five hand-designed families, one noise level (0.05), one observation protocol (30 points, always including t = 0, forecast window 7 to 10).
3. **Only naive baselines.** Hold-last, constant velocity, linear interpolation. No per-family curve fitting, no sequence-model (RNN / ODE-RNN) baseline.
4. **Forecasting is not directly trained.** The loss supervises only the 30 observed points; the option to supervise the future (`full_clean`) was never run, so the link between this and poor extrapolation is a hypothesis.
5. **Not an exact replication.** Different data, solver, objective details; the paper's numbers were not reproduced; the classification and CNF experiments are not in this repository.
6. **Solver.** Fixed-step rk4 (3/8 rule) with linear interpolation to off-grid query times; measured effect on RMSE about 2e-4 (negligible here), but outputs are not fourth-order accurate between steps.
7. **Not fully converged.** Selected epochs are 294-299 of 300 and the validation loss was still falling.
8. **Observation count.** Models only work near the 30 observations they were trained with.
9. **Confounds.** In the capacity study the data stream was not controlled; in all size studies latent dimension and GRU width are changed together or separately but only at two levels each; the five-family vs three-family comparison also changes training size and steps (33,000 vs 19,800).
10. **Zero-shot evidence is narrow.** One frozen model for figure-eight and sinusoid; no leave-one-family-out experiment.
11. **No improved forecasting method** was developed or tested.
12. **GeoLife.** One dataset (mostly Beijing), one run per model, licence prevents sharing data or trained weights.

## Future work (recommended, not executed), in priority order

| Priority | Task | Why | Needs training? | Cost estimate (measured per-run figures) |
|---|---|---|---|---|
| 1 | **Multi-seed repeat of the 2x2 ablation** (at least A and D; B and C if affordable) with seeds 123, 456, 789, 2026, setting `--seed` and `--data-seed` per seed | tells whether the forecast differences are real or seed noise | yes | about 80 min and 0.3 MB per run; A and D = 8 runs; all four = 16 runs; four runs in parallel on 16 of 32 cores were measured, eight in parallel not yet |
| 2 | Stronger baselines: a per-family parametric curve fit extrapolated forward; a sequence-model baseline (RNN / ODE-RNN) | currently only naive baselines | parametric: no; neural: yes | parametric: minutes |
| 3 | Leave-one-family-out generalisation (train on four families, test on the fifth) with several seeds | the cross-family claim rests on one frozen model | yes | five hold-out choices x seeds x 80 min |
| 4 | Test the observed-points-only objective hypothesis (`--supervision full_clean`, clearly labelled as a supervised diagnostic) | tests whether the objective limits extrapolation | yes | 2 to 4 runs |
| 5 | If a mechanism is found, evaluate one candidate forecasting improvement under the same protocol, multi-seed, against the same baselines | only then can an improvement be claimed | yes | depends |
| 6 | Horizon-resolved analysis for all models and the five-family data (no training) | already done for the 2x2 overall; extend per family | no | minutes |
| 7 | Literature review of work after 2018 to check whether cross-family latent-ODE forecasting is an open gap | gap statement is NOT VERIFIED | no | - |
| 8 | Complete the GeoLife write-up (report sections 9 and 11); other GPS datasets | partly done | maybe | - |

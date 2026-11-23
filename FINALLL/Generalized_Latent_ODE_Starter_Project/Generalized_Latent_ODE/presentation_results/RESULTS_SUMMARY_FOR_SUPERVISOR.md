# Results summary for the supervisor (simple English)

Date: 2026-11-21. Everything below is taken from results already saved in the project (see `RESULTS_SOURCE_INDEX.md`). Nothing was re-run. **Almost all results are from a single training run per model unless marked "5 runs".** A single run can be lucky or unlucky, so single-run differences of 20% or less should not be treated as findings.

## The project in five sentences
1. We use a *Latent Neural ODE*: a neural network that looks at a few noisy points of a 2-D path, turns them into a hidden starting state, evolves that state forward in continuous time with a learned rule, and draws the path.
2. It is an adapted re-implementation and extension of Chen et al. (2018), **not** an exact replication and **not** a new or improved forecasting model.
3. We tested it on five path types (circle, spiral, ellipse, figure-eight, sinusoid) with one shared model and no shape label.
4. We asked: does one model for many path types lose accuracy; does it generalise to path types it never saw; and how does model size matter?
5. Every model sees only 30 noisy points from times 0 to 7 and must (a) fill in the gaps inside that period ("reconstruction") and (b) predict times 7 to 10 ("forecast").

## What we found, step by step

**Step 1. One shared model works for all five path types.** It forecasts far better than the simple baselines (cell A forecast error 0.3998 against 1.0238 for staying at the last point). But the smallest models reconstruct no better than joining the dots with straight lines (cell A 0.0882 against 0.0672). *(single run)*

**Step 2. A shared model is less accurate than specialist models.** For circles and spirals, one model trained on both has 85% (circle) and 43% (spiral) higher forecast error than separate models. *(5 shared runs vs 3 specialist runs; clear for circles, weaker for spirals.)*

**Step 3. Unseen path types are not handled.** The three-family model, tested on figure-eights and sinusoids it never saw, forecasts worse than just staying at the last point (1.1174 vs 0.9610 and 0.9734 vs 0.6248). Models trained without ellipses draw ellipses as near-circles (forecast 0.32, reconstruction 0.21, against 0.06 for joining the dots). *(ellipse: 5 runs; figure-eight and sinusoid: 1 run)*

**Step 4. Bigger encoders reconstruct better (single-run result).** In the controlled 2x2 experiment reconstruction error falls by 48% (latent 16), 30% (GRU 128) and 53% (both), and it is lower in every family.

**Step 5. Forecasting with bigger encoders is mixed.** Over the whole forecast window, latent 16 forecasts 30% worse and both together 48% worse, but widening the GRU alone is not clearly different (6% higher, not significant). The larger models are actually **better** just after t = 7 and become worse after about t = 8 to 8.6. *(single run; the differences are as large as the variation you get by simply retraining the same architecture, about 21 to 22%, so this is a hint, not a finding.)*

**Step 6. Real GPS data defeats the synthetic-trained model.** Straight-line interpolation has a median error of 7.1 m; our synthetic model 126 m. Training on real GPS data brings it to 61 m (scratch) or 47 m (fine-tuned), still about 6 to 8 times worse than the straight line. *(single run)*

## Answers to the ten questions

1. **What experiments have we completed?** Failure diagnosis and Model D; 5-seed reproducibility; shared vs specialist; three families and unseen ellipses; dynamics conditioning (5 seeds); observation-count sensitivity; GeoLife transfer and GeoLife training; the five-family dataset with zero-shot test; capacity scaling; the controlled 2x2 ablation; and an independent audit.
2. **Most important numbers.** Shared vs specialist forecast: circle 0.161 vs 0.087, spiral 0.284 vs 0.198. Unseen families (3-family model): figure-eight 1.1174, sinusoid 0.9734 vs hold-last 0.9610, 0.6248. Ablation (A, B, C, D): reconstruction 0.0882, 0.0462, 0.0619, 0.0412; forecast 0.3998, 0.5199, 0.4248, 0.5936. GeoLife (30 obs): linear 7.1 m vs synthetic model 126.4 m.
3. **Best at reconstruction?** On the five-family data the larger encoders: cell D 0.0412 (and the earlier 16/128 run 0.0399, practically the same recipe), against 0.0882 for the smallest model and 0.0672 for linear interpolation. On circle and spiral data the specialists are slightly better than the shared model (circle 0.0457 vs 0.0532). (One training run each, except the specialist comparison.)
4. **Best at forecasting?** The smallest models: on the five-family data the earlier 8/64 run (0.3265) and its controlled rerun A (0.3998) are better over the full window than the larger ones; on circle and spiral the specialists (0.087, 0.198). The larger models are better only in the first time unit after t = 7. The runs are single, so the ranking among the smaller variants (8/64 against 8/128) is not firm.
5. **Is the shared model better than specialists?** No. It is worse at forecasting (clearly for circle) and slightly worse at reconstruction; its advantage is convenience (one model, no labels).
6. **Does it generalise to unseen families?** No for figure-eight and sinusoid (worse than simple baselines); only partially for ellipses (direction and rough forecast right, shape wrong, reconstruction poor); no for real GPS.
7. **What does the controlled ablation demonstrate?** That bigger encoders reconstruct better, and that in one set of runs latent size is associated with worse late-horizon forecasts while GRU width alone is not clearly different. It does **not** demonstrate that capacity hurts forecasting in general or why.
8. **Have we developed a better forecasting model?** **No.** We re-implemented and analysed the Latent ODE; no new method has been proposed or tested, and the larger variants did not forecast better over the full window.
9. **What is strong enough to present?** The verified implementation and data; the shared-vs-specialist cost; the failure to generalise to unseen families and to real GPS; reconstruction gains from larger encoders; the ablation as an exploratory single-run result with its caveats.
10. **Most important remaining questions.** Do the ablation effects repeat across seeds? Why do shared models forecast worse (and can the gap be closed)? Does a model trained on four families handle the fifth? Is the observed-points-only training objective limiting forecasts? How do stronger baselines (fit a curve per family, a sequence model) compare?

## Key tables

**Five-family models: mean error (one training run each).**

| Family | Linear interp. (recon) | A recon | D recon | Hold-last (forecast) | A forecast | D forecast |
|---|---:|---:|---:|---:|---:|---:|
| circle | 0.0699 | 0.0607 | 0.0369 | 1.1370 | 0.2676 | 0.5182 |
| spiral | 0.0653 | 0.0883 | 0.0434 | 1.2763 | 0.4915 | 0.6605 |
| ellipse | 0.0637 | 0.0946 | 0.0408 | 1.1200 | 0.3916 | 0.5240 |
| figure-eight | 0.0917 | 0.1364 | 0.0568 | 0.9610 | 0.4740 | 0.4845 |
| sinusoid | 0.0452 | 0.0607 | 0.0282 | 0.6248 | 0.3744 | 0.7808 |
| all five | 0.0672 | 0.0882 | 0.0412 | 1.0238 | 0.3998 | 0.5936 |

**Shared vs specialist (circle + spiral data).**

| Family | Forecast: shared (5 runs) | Forecast: specialist (3 runs) | Difference | Statistical evidence |
|---|---:|---:|---:|---|
| circle | 0.161 +/- 0.037 | 0.087 +/- 0.006 | +85% | all 5 shared runs worse than all 3 specialist runs; Welch p = 0.0099 (0.04 after Bonferroni) |
| spiral | 0.284 +/- 0.067 | 0.198 +/- 0.009 | +43% | all 5 shared runs worse than all 3 specialist runs; Welch p = 0.045 (0.18 after Bonferroni) |

**Unseen families (three-family model, one run) and ellipses (two-family models, 5 runs).**

| Unseen family | Forecast | Hold-last | Reconstruction | Linear interp. |
|---|---:|---:|---:|---:|
| figure-eight | 1.1174 | 0.9610 | 0.7321 | 0.0917 |
| sinusoid | 0.9734 | 0.6248 | 0.3441 | 0.0452 |
| ellipse (5 runs, mean) | 0.3212 | 1.1213 | 0.2096 | 0.0625 |

**Controlled 2x2 ablation (one run each).**

| Model | Latent / GRU | Reconstruction | Forecast | Forecast vs A |
|---|---|---:|---:|---:|
| A | 8 / 64 | 0.0882 | 0.3998 | n/a |
| B | 16 / 64 | 0.0462 | 0.5199 | +30.1% |
| C | 8 / 128 | 0.0619 | 0.4248 | +6.3% |
| D | 16 / 128 | 0.0412 | 0.5936 | +48.5% |

## What to say, and what not to say
**Safe to say:** the model works; one shared model costs forecast accuracy for circle and spiral; it does not generalise to unseen path types or to real GPS; bigger encoders reconstruct better; the full-window forecast difference between model sizes is exploratory, appears late in the horizon, and is the same size as seed-to-seed noise.
**Do not say:** that capacity hurts forecasting in general; that we found why; that we replicated the paper; that we built a better forecasting model; that the five-family model shows generalisation (it saw all five families).

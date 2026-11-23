# Presentation content: results slides and speaking script

Prepared for the meeting with the supervisor ("Ma'am"). 13 slides; figures are in `presentation_results/figures/` (new, drawn from saved data) or the existing project figures named below. Numbers are exact (4 decimals in tables, rounded in speech). **Say "single run" whenever a result comes from one training run.**

## Slide 1: What we built and what we asked

- A Latent Neural ODE: noisy points in, hidden start state, learned continuous-time rule, curve out.
- An adapted re-implementation and extension of Chen et al. (2018), not an exact replication and not a new method.
- Five path types (circle, spiral, ellipse, figure-eight, sinusoid), one shared model, no shape label.
- Each model sees 30 noisy points from t = 0 to 7, then must reconstruct the gaps and forecast t = 7 to 10.

**Figure / table:** Existing figure `results/five_shapes/figures/five_shapes_gallery.png` (the five families).

**Exact numbers:** 24,602 parameters in the baseline model; 1,000 trajectories per family; test set 750 trajectories (150 per family).

**What it means (plain English):** We are testing how well one learned model copes with different kinds of motion, and how much model size matters.

**Speaking script:** "Our project builds on the Neural ODE paper. We did not copy its experiment: we used our own synthetic paths, five types in all. The model sees only thirty noisy points from the first seven time units. We measure two things: how well it fills in the gaps, and how well it predicts the future from seven to ten."

**Limitation to state:** This is a re-implementation and extension, not a replication, and we have not developed a new method.

## Slide 2: How we measure success: simple baselines

- Reconstruction error = distance from the true path at unobserved times inside t <= 7 (lower is better).
- Forecast error = distance from the true path for 7 < t <= 10 (lower is better).
- Baselines: linear interpolation (join the dots), hold-last (stay at the last point), constant velocity (keep moving at the last speed).
- A learned model is only useful if it beats these.

**Figure / table:** Table (below).

**Exact numbers:** All-five-families baselines: linear interpolation 0.0672 (reconstruction); hold-last 1.0238; constant velocity 2.1474 (forecast).

**What it means (plain English):** Straight lines are already quite good for filling gaps, so a neural model must do better than 0.067 to be worth it; for forecasts, staying still gives about 1.0.

**Speaking script:** "Before showing results I need to explain the yardsticks. Our error measure is the average distance from the true path. For gaps inside the observed period, the simple comparison is joining the dots with straight lines. For the future, the simple comparisons are staying at the last point, or continuing at the last speed."

**Limitation to state:** These baselines are deliberately simple; stronger classical baselines (a curve fitted per family) were not tested.

| Baseline | Task | Mean error, all five families |
|---|---|---:|
| Linear interpolation | reconstruction | 0.0672 |
| Hold-last | forecast | 1.0238 |
| Constant velocity | forecast | 2.1474 |

## Slide 3: Result 1: reconstruction by family

- Larger encoders (B, C, D) reconstruct better than the smallest model A in every family.
- A is worse than joining the dots on four of five families; B and D beat it on all five.
- Sinusoid is the easiest family to reconstruct (lowest in 6 of 7 models); figure-eight is the hardest (6 of 7).
- All numbers are single training runs.

**Figure / table:** `figures/fig1_reconstruction_by_family.png`

**Exact numbers:** Overall: A 0.0882, B 0.0462, C 0.0619, D 0.0412; linear interpolation 0.0672. Figure-eight: A 0.1364, D 0.0568.

**What it means (plain English):** The bigger encoders fit the observed part much more tightly. The figure-eight is hard because it turns sharply and crosses itself.

**Speaking script:** "On this slide, lower bars are better and the black dash is straight-line interpolation. The smallest model, in blue, is worse than the dash in most families. Models B and D, in orange and yellow, are clearly below it in every family; C, in green, is about level with it. The wave-shaped sinusoid is the easiest family; the figure-eight is the hardest."

**Limitation to state:** Single training run per model; reconstruction is what the training loss optimises directly, so this improvement is expected.

## Slide 4: Result 2: forecasting by family

- All models forecast far better than hold-last overall.
- The hardest family to forecast depends on size: spiral for the smallest models, sinusoid for the larger ones.
- For B and D the sinusoid forecast is worse than the hold-last baseline (0.8602 and 0.7808 vs 0.6248).
- Circle and ellipse are generally the easiest.

**Figure / table:** `figures/fig2_forecast_by_family.png`

**Exact numbers:** Overall forecast: A 0.3998, B 0.5199, C 0.4248, D 0.5936; hold-last 1.0238. Circle A 0.2676; spiral A 0.4915.

**What it means (plain English):** The model does not perform equally on all families: forecasting a wave drifting sideways is hard for the bigger models, and spirals are hard for the small ones.

**Speaking script:** "For forecasting, the models are well below the staying-still baseline in almost every case. The one clear exception is the sinusoid, where two of the larger models do worse than just staying at the last point."

**Limitation to state:** Single run per model; the constant-velocity baseline (2.0 to 2.6) is off the chart.

## Slide 5: Result 3: one shared model vs specialist models (circle + spiral)

- Shared model (5 runs) vs specialists (3 runs): circle forecast 0.161 vs 0.087 (+85%); spiral 0.284 vs 0.198 (+43%).
- Every shared run was worse than every specialist run.
- Reconstruction is close (+16% and +8%).
- Shared forecasts also vary more from run to run.

**Figure / table:** `figures/fig3_shared_vs_specialist.png`

**Exact numbers:** Welch p for circle forecast = 0.0099 (0.04 after Bonferroni); spiral forecast p = 0.045 (0.18 after Bonferroni, not significant). SD across runs: shared 0.037 / 0.067; specialists 0.006 / 0.009.

**What it means (plain English):** Using one model for several families is convenient but costs forecast accuracy: specialists are better.

**Speaking script:** "This is the main multi-run experiment. Each dot is one training run. The blue shared model is clearly worse at forecasting than the orange specialists, and its runs vary more. For circles this holds even after correcting for multiple tests; for spirals the difference is in the same direction but weaker."

**Limitation to state:** Only two families have specialists, and only three specialist runs exist.

## Slide 6: Result 4: unseen families: no zero-shot generalisation

- Three-family model on unseen figure-eight / sinusoid forecasts worse than hold-last (1.1174 vs 0.9610; 0.9734 vs 0.6248).
- Reconstruction is far worse than joining the dots (0.7321 vs 0.0917; 0.3441 vs 0.0452).
- The five-family model has seen all five, so its results are not zero-shot.
- On the three original families the five-family model is worse than the three-family one (forecast +85%, +47%, +38%).

**Figure / table:** `figures/fig4_cross_family_generalisation.png`

**Exact numbers:** Five-family model forecast: circle 0.2594, spiral 0.4270, ellipse 0.2609, figure-eight 0.2992, sinusoid 0.3861. Training steps 19,800 (3-family) vs 33,000 (5-family).

**What it means (plain English):** A model trained on loops and spirals cannot suddenly draw figure-eights or waves. And adding two families to the training set cost some accuracy on the old ones, though training size and steps also changed.

**Speaking script:** "In the shaded rows the blue filled dot is the three-family model on families it never saw. It is far from the baselines in the wrong direction. The hollow dot is the five-family model, which did see those families, so it is not a generalisation test."

**Limitation to state:** One frozen model, one seed; no leave-one-family-out experiment.

## Slide 7: Result 5: unseen ellipses: partial transfer (5 runs)

- Models trained on circle + spiral only, tested on ellipses: forecast 0.321 +/- 0.027, much better than hold-last (1.12).
- Reconstruction 0.210, more than 3 times worse than joining the dots (0.063).
- Rotation direction correct for 100% of ellipses and shape wrong (seed-42 model): predicted axis ratio 0.92 vs true 0.59.
- An ellipse is a stretched circle, so this is the easiest unseen family.

**Figure / table:** Table (below).

**Exact numbers:** In-training reference (three-family model): reconstruction 0.089, forecast 0.185.

**What it means (plain English):** The model keeps the rotation and roughly the size but turns ellipses into near-circles.

**Speaking script:** "Ellipses are the friendliest unseen family. The models still get the direction right and forecast roughly, but they round the shape toward a circle and fail at filling gaps accurately. This is partial transfer, not success."

**Limitation to state:** Based on 5 two-family runs; ellipses are geometrically close to circles.

| Model on ellipses | Reconstruction | Forecast |
|---|---:|---:|
| Two-family models (5 runs), unseen | 0.2096 | 0.3212 |
| Three-family model, seen in training | 0.0887 | 0.1850 |
| Linear interpolation / hold-last | 0.0625 | 1.1213 |

## Slide 8: Result 6: controlled 2x2 ablation: reconstruction and forecast

- Four models: latent 8 or 16, GRU 64 or 128; same data, recipe and data stream; one run each.
- Reconstruction improves: B -48%, C -30%, D -53% vs A.
- Full-window forecast: B +30%, D +48%; C +6% (not significant).
- Widening the GRU alone made no clearly distinguishable forecast difference.

**Figure / table:** `figures/fig5_ablation_overall.png`

**Exact numbers:** Parameters: A 24,602, B 27,186, C 63,642, D 67,250. C vs A forecast: mean difference +0.0250, Holm p = 0.18; better on 48% of trajectories. Interaction (forecast) +0.049, p = 0.0018.

**What it means (plain English):** Bigger encoders fit the observed part better. For forecasting, increasing latent size is associated with worse full-window results; widening the GRU alone is not clearly different.

**Speaking script:** "This is our most controlled experiment: four models differing in only two settings. Reconstruction improves with size. For forecasting, only the latent-size changes look worse. I want to stress that each bar is a single training run."

**Limitation to state:** Single seed: effects of +6% to +48% are the same size as run-to-run variation (about 21 to 22% for the same architecture).

## Slide 9: Result 7: forecast error over time

- Larger models are better just after t = 7, worse later.
- They cross above model A at t = 8.09 (B), 8.64 (C), 7.94 (D).
- Window 7-8: A 0.263, B 0.217, C 0.176, D 0.233; window 9-10: A 0.482, B 0.704, C 0.595, D 0.810.
- So "worse forecasting" depends on the horizon.

**Figure / table:** `figures/fig6_forecast_error_vs_time.png`

**Exact numbers:** RMSE across the 750 test trajectories at t = 10: A 0.623, B 0.912, C 0.828, D 1.012.

**What it means (plain English):** Errors always grow with time, but they grow faster for the models with larger latent space.

**Speaking script:** "This plot is why I am careful with the word "worse". Right after the forecast starts, the bigger models are better. After about t equals eight they overtake model A and keep growing. A shorter forecast window would give the opposite conclusion."

**Limitation to state:** Single run; the cause (for example the training objective only sees observed points) is a hypothesis we have not tested.

## Slide 10: Why we are careful: effect sizes vs run-to-run variation

- Same architecture, two training runs: forecast error differs by +22% and +21%.
- Five seeds, same recipe (two-family data): forecast 0.169 to 0.283, CV about 23%.
- Ablation forecast effects: +30%, +6%, +48%.
- So single-run forecast differences of this size are not findings yet.

**Figure / table:** `figures/fig7_effects_vs_seed_variation.png`

**Exact numbers:** Two-family five-seed mean 0.2194 +/- 0.0502 (95% CI [0.157, 0.282]).

**What it means (plain English):** Retraining the same model gives noticeably different forecasts, so a difference between two single runs may just be luck.

**Speaking script:** "This slide puts the ablation effects on the same scale as the variation we see when we simply retrain the same recipe. They are comparable. That is why I call the ablation exploratory and why multi-seed runs are the next step."

**Limitation to state:** The three groups come from different experiments and are placed on one axis only to show scale.

## Slide 11: Real-world test: GPS data (GeoLife)

- Linear interpolation 7.1 m vs frozen synthetic model 126.4 m: it does not transfer.
- Trained on real GPS: scratch 60.6 m, fine-tuned 47.1 m: better, but still 6 to 8 times worse than the straight line.
- Fine-tuning from synthetic weights beats scratch only at 30 observations (ratio 0.78); worse at 10, 20, 40, 50.
- Forecasts are worse than hold-last and constant velocity.

**Figure / table:** `figures/fig8_geolife_transfer.png`

**Exact numbers:** Median error (30 observations, 1,000 segments, 97 users): cubic 5.56 m, PCHIP 5.62 m. Fine-tuned / scratch ratios: 10 obs 1.65, 20 1.22, 30 0.78, 40 1.29, 50 1.93.

**What it means (plain English):** Real movement is not made of the loops our synthetic data teaches; simple interpolation wins easily.

**Speaking script:** "We also asked whether a model trained on synthetic shapes helps with real GPS tracks. It does not: straight lines are much better. Training on real data helps but does not close the gap. Synthetic pre-training helped only at the thirty observations used in training."

**Limitation to state:** One dataset (mostly Beijing), one run per model; the fine-tuned versus scratch comparison was computed from the saved per-segment results.

## Slide 12: What our experiments show, and do not show

- Shown: the implementation and data are valid; shared models cost forecast accuracy; no zero-shot generalisation; larger encoders reconstruct better.
- Exploratory only: forecasts vs model size (single run, late-horizon, same size as run-to-run noise).
- Not shown: a better forecasting model, a cause for the forecast differences, results across seeds for the five-family work.
- The project is an adapted re-implementation and experimental extension, not a replication or a new method.

**Figure / table:** Table (below).

**Exact numbers:** See table.

**What it means (plain English):** We have a solid, verified analysis framework and several clear negative results; the main positive result is reconstruction from larger encoders.

**Speaking script:** "To summarise: we can clearly say what does not work: sharing a model costs forecast accuracy, and unseen families and real GPS are not handled. We cannot yet say why forecasting gets worse with size; that needs multiple seeds."

**Limitation to state:** All claims are limited to this synthetic data and these training runs.

| Question | Answer | Evidence |
|---|---|---|
| Shared better than specialist? | No (forecast) | 5 vs 3 runs |
| Generalises to unseen families? | No / partial for ellipses | 5 runs (ellipse), 1 run (others) |
| Larger encoders improve reconstruction? | Yes | single run, consistent |
| Larger encoders worsen forecasts? | Exploratory, late horizon only | single run |
| Better forecasting model developed? | No | n/a |

## Slide 13: Next steps

- Repeat the 2x2 ablation with several seeds (at least A and D, four more seeds each).
- Add stronger baselines: a curve fitted per family, and a sequence-model baseline.
- Leave-one-family-out generalisation tests with several seeds.
- Test whether training only on observed points limits forecasts (a labelled diagnostic), before proposing any improvement.

**Figure / table:** Plan list (no figure).

**Exact numbers:** Each repeat is about 80 minutes of CPU training per run, about 0.3 MB per checkpoint.

**What it means (plain English):** These are the experiments that would turn our exploratory observations into conclusions.

**Speaking script:** "The most important next step is repeating the controlled ablation with several seeds, because that tells us whether the forecast differences are real. After that, stronger baselines and leave-one-family-out tests."

**Limitation to state:** None of these experiments has been run.


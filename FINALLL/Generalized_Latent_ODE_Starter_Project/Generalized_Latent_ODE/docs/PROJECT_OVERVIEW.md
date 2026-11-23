# Project overview

**Generalized Latent Neural ODE: cross-family generalisation and forecasting in Latent Neural Ordinary Differential Equations.**

The project studies one shared Latent Neural ODE that reads sparse, noisy, irregularly timed 2-D observations (t <= 7) and must reconstruct the path and forecast it (7 < t <= 10) without being told which trajectory family it is. It then asks how far this goes: unseen families, model size, specialist models, conditioning, and real GPS data (GeoLife).

It is an **adapted re-implementation and experimental extension** of Chen et al. (2018), not an exact replication, and it has **not** produced an improved forecasting model. The honest findings are mixed: larger or shared models reconstruct well, but unified models forecast worse than specialists, unseen families are not handled well, the synthetic model does not transfer to GeoLife, and most size-related findings are single-seed.

Where to read next: [README](../README.md) (summary, installation, quick start) · [ARCHITECTURE](ARCHITECTURE.md) · [DATASETS](DATASETS.md) · [METHODOLOGY](METHODOLOGY.md) · [EXPERIMENTS](EXPERIMENTS.md) · [RESULTS](RESULTS.md) · [INFERENCE_GUIDE](INFERENCE_GUIDE.md) · [REPRODUCIBILITY](REPRODUCIBILITY.md) · [RESEARCH_LIMITATIONS](RESEARCH_LIMITATIONS.md) · [REFERENCES](REFERENCES.md).

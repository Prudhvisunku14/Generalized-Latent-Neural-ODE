# Which upstream code components informed this project?

| Upstream file or concept | Public source | Our independent module | Difference |
|---|---|---|---|
| `RecognitionRNN` | https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py | `src/glo/model.py::RecognitionGRU` | Uses a GRU, includes normalized timestamps and reverse inter-observation gaps. |
| `LatentODEfunc` | Same file | `src/glo/model.py::LatentDynamics` | One hidden layer for the baseline, plus optional fixed conditioning state. |
| `Decoder` | Same file | `src/glo/model.py::LatentODE.decoder` | Continuous two-dimensional Gaussian-mean predictions. |
| `odeint` / `odeint_adjoint` | https://github.com/rtqichen/torchdiffeq/blob/master/README.md | `src/glo/solver.py::integrate` | Wraps official solver; local *test-only* RK4 fallback if dependency is unavailable. |
| `generate_spiral2d` | https://github.com/rtqichen/torchdiffeq/blob/master/examples/latent_ode.py | `src/glo/data.py::sample_trajectory` | Our new dataset uses logarithmic spirals, variable centers/frequencies/directions/parameters, split hygiene and additional shape families. |
| Latent ODE training with ODE-RNN encoder | https://github.com/YuliaRubanova/latent_ode | **Not used as our baseline** | Follow-up 2019 research, relevant for a later encoder comparison. |

The source code in `src/glo` was written for this project. It uses standard mathematical formulas and the public solver API, but we did **not** wholesale copy or vendor upstream Python source files. Consult the original repositories for their complete licensed implementations.

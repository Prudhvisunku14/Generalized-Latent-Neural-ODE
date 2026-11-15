# Sources used to construct this reproduction

1. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS 31.
   - https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html

2. Authors' `torchdiffeq` repository.
   - https://github.com/rtqichen/torchdiffeq
   - Initial public release commit inspected: `810fda4e53d0746a65442d07f5d865cc8ee386f3`
   - The release README states that its MNIST script is a reproduction and uses a different ODE solver library/method from the original paper experiment.

3. Historical latent-ODE 2-D spiral example added to the authors' `torchdiffeq` repository.
   - Commit inspected: `530b8b6a379cc105f8e48220ed4195c4ae5e2b01`

4. Rezende, D. J. & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*.
   - Used for the standard planar-flow target potentials underlying the Figure-4 reconstruction.

This folder contains a fresh, consolidated reproduction implementation rather than a verbatim copy of those repositories.

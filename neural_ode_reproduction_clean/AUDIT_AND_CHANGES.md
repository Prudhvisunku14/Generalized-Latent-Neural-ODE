# Audit of the supplied ZIP and corrections made

The supplied project already had useful model code, checkpointing and paper notes.  It did **not** need to be discarded completely.  This rebuild preserves the sound mathematical components while correcting pipeline problems and removing generated clutter.

## Major corrections

1. **GPU selection fixed**
   - Previous `get_device()` always called `torch.cuda.set_device(0)`, even when a user requested another GPU.
   - The rebuilt version respects `cuda:N`, adds `--gpu N`, prints the selected GPU/VRAM, and supports `--require-cuda` so a long run cannot silently fall back to CPU.

2. **GPU performance path cleaned up**
   - Non-blocking host-to-GPU transfers, pinned DataLoader memory, persistent workers and cuDNN benchmarking are used where appropriate.
   - FP32 is retained for ODE/CNF/Latent-ODE numerical reliability.

3. **Experiment 1 now includes the 1-layer MLP**
   - The supplied code only trained ResNet/RK-Net/ODE-Net although the requested comparison includes the 1-layer 300-unit MLP.
   - The paper's MLP value remains clearly marked as an external LeCun reference.

4. **MNIST ODE dynamics aligned with the authors' public reproduction**
   - The supplied code inserted time as an extra convolution channel via `ConcatConv2d`.
   - The public `odenet_mnist.py` reproduction uses ordinary 3x3 convolutions in `ODEFunc`; the extra time channel was removed.

5. **Epoch loss logging corrected**
   - The supplied MNIST script printed/saved only the final mini-batch loss of an epoch.
   - It now saves the sample-weighted mean training loss over the full epoch.

6. **Forward and backward NFE are both recorded**
   - ODE-Net training now records mean forward and backward NFE separately.
   - Figure 3 also saves every raw tolerance/batch measurement to CSV.

7. **Memory/time results added**
   - Every epoch/logging epoch reports elapsed time, total elapsed time, ETA and peak allocated GPU memory.
   - Experiment 1's final comparison table contains both theoretical paper memory complexity and measured peak GPU memory.

8. **Experiment 2 output pipeline completed**
   - Figure 4 losses are saved to `figure4_losses.csv`.
   - The unused `viz_samples` option was removed.
   - Static outputs are saved as 600-dpi PNG plus PDF.
   - Two Circles/Two Moons transformations are exported as trained-model NF-vs-CNF GIFs.
   - Stage samples are retained in compressed `.npz` files.

9. **Experiment 3 evaluation fairness fixed**
   - The supplied code evaluated the RNN and Latent ODE on *different randomly sampled observation indices* because two different RNG seeds were used.
   - Both models now use exactly the same irregular observation mask for each test trajectory.

10. **Experiment 3 irregular sampling improved**
    - The supplied training code used one common set of observation indices for an entire batch.
    - The rebuild samples observation indices independently for every trajectory in a batch.

11. **Unsupported hard-coded time-gap scaling removed from the Latent ODE recognition network**
    - The supplied recognition RNN divided time gaps by a hard-coded `10 * 0.01`, while the actual local time grid was different.
    - The paper states that the recognition RNN consumes observations backwards; the rebuilt recognition RNN therefore follows that stated architecture without the unsupported scaling.
    - The RNN baseline can still use time gaps, consistent with the paper's discussion of a time-aware RNN baseline. Use `--rnn-no-dt` to disable this.

12. **Train/test split made explicit and shuffled**
    - The rebuild shuffles trajectories with a fixed seed before the 80/20 split so the split does not depend on generator order.

13. **30/50/100 observation visualizations completed**
    - The supplied Exp 3 code only generated a Figure-8-style visualization for the 30-observation case.
    - The rebuild creates separate publication-quality comparisons for 30, 50 and 100 observations, plus latent-space PNG/PDF/GIF and a 30-observation trajectory GIF.

14. **Checkpoint resume made more robust**
    - Optimizer/RNG state are restored even from a checkpoint previously marked complete, allowing a user to increase the epoch/iteration budget and continue.
    - `--fresh` explicitly discards matching checkpoints when a clean restart is desired.

15. **Repository cleaned**
    - Removed bundled `__pycache__`, `.pytest_cache`, generated smoke results, checkpoints and downloaded MNIST data from the deliverable.
    - Added `.gitignore` so those files do not re-enter the source archive.

## Assumptions that remain

Some details are not specified by the 2018 paper and therefore cannot be "corrected" to a uniquely known value: RK-Net fixed step size, Figure 4 learning rates/batch size/integration horizon, Figure 5 iteration budget/noise/scale, and the exact Latent-ODE train/test split and synthetic trajectory implementation.  These are kept explicit in code/comments and in `REPRODUCIBILITY_NOTES.md` rather than being presented as paper facts.

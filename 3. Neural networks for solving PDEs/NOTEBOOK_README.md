# Neural PDE Laboratory

Open [Neural_PDE_Laboratory.ipynb](Neural_PDE_Laboratory.ipynb) to work through the experiments. It includes saved outputs and runs on its own once the dependencies are installed. To browse the figures, download `notebook_outputs/` and open [index.html](notebook_outputs/index.html) in a browser.

The thirteen experiments include explanations and suggestions for what to try next:

1. A vibrating 2D membrane: modal interference, conserved energy, and an animated 3D surface.
2. A 1D Poisson problem: strong PINN versus Deep Ritz, with a finite-difference reference.
3. Hard versus soft boundary conditions and a penalty-weight sweep.
4. A 2D Poisson PINN, with independent error/residual maps and a 3D explorer.
5. Tanh nonattainment and a continuous-residual counterexample.
6. Inverse diffusion: sparse sensor placement, coefficient recovery, uncertainty, and an identifiability valley.
7. High-dimensional Monte Carlo energy integration, checked against an exact RMS formula.
8. A neural Poisson solution on an annulus, with area-uniform sampling and exact boundary enforcement.
9. Gray–Scott reaction–diffusion patterns, with a classical solver, time-step diagnostic, and animated 3D concentration landscape.
10. Multiscale Poisson: plain tanh versus Fourier-feature inputs, with modal recovery during training.
11. Residual-adaptive collocation: importance weighting, localization, and a comparison at a fixed point/iteration budget.
12. A space–time heat PINN: initial and boundary enforcement, diffusion of Fourier modes, and animated profiles.
13. A parameter-conditioned PINN: one trained network for a reaction–diffusion family, compared with a finite-difference reference.

## Run it

Use Python 3.11 or newer. From this folder (`3. Neural networks for solving PDEs/`):

```sh
python -m pip install -r experiments/requirements-notebook.txt
python -m jupyter lab Neural_PDE_Laboratory.ipynb
```

Select **Restart Kernel and Run All Cells**. The initial cells contain the imports, shared plotting helpers, seed, and training budgets. The supplied notebook already has outputs. Trust it in Jupyter to enable embedded JavaScript; standalone HTML explorers also work offline in a browser.

The supplied run took about 104 seconds on the machine used to prepare it, including figure exports. It uses CPU/float64 and short training runs. The measured total runtime is recorded in `notebook_outputs/run_summary.json` and printed in the final cell. Runtime varies by machine. `RUN_MODE = "thorough"` increases budgets in the chapters that provide this option; each experiment lists its optimizer and sampling settings. All experiments compute their own inputs and results, so the notebook can be copied and run independently of the slide source and experiment scripts.

## Outputs

Running the notebook writes to `notebook_outputs/` in the kernel's working directory:

- 16 static figure sets, each as PNG and vector PDF.
- Seven self-contained interactive HTML explorers for membrane motion, Poisson fields, the annulus, reaction–diffusion, modal learning, heat flow, and the parameterized PDE family.
- `index.html`, a visual gallery linking these outputs.
- `run_summary.json`, with versions, seeds, budgets, and measured errors.
- `computed_fields.npz`, with numerical fields and diagnostics.

Read the axis labels on the 3D plots: height represents a field value, and some plots use time or a PDE parameter as an axis. Each experiment explains whether it uses an analytic solution, a neural network, a classical solver, or a sampling calculation. Forward neural solves use the forcing and boundary conditions for training and the exact solution for error checks.

To edit the notebook, change the source cells in `experiments/notebook_parts/`, then rebuild and run it in a fresh kernel:

```sh
python experiments/build_notebook.py --execute
```

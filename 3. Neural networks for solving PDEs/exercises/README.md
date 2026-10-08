# Neural PDE practical exercises

Six self-contained notebooks turn the chapter's six sections into practical work.
Each starts with a working baseline, then asks you to change a setting, compare
results, and explain what the diagnostics establish. Saved plots and small output
tables make the baseline readable on GitHub.

## Choose an exercise

| Course section | Notebook | Expected deliverable |
|---|---|---|
| From data to PDE solvers | [01 · From observations to a PDE loss](Neural_PDE_Exercise_01_data_to_pde.ipynb) | a noise/weight comparison, an annotated fit, and an explanation of the role of the PDE |
| Strong-form PINNs | [02 · Collocation and an unseen residual](Neural_PDE_Exercise_02_strong_form.ipynb) | a sampling comparison and an explicit residual that collocation misses |
| Network design and boundary constraints | [03 · Boundary constraints and smoothness](Neural_PDE_Exercise_03_boundaries_and_design.ipynb) | a boundary-penalty sweep and a discussion of activation regularity |
| Variational and weak methods | [04 · Energy and weak residuals](Neural_PDE_Exercise_04_variational_and_weak.ipynb) | a study of test-space size and quadrature resolution, with independent errors |
| Hybrid PDE–data methods | [05 · Inverse diffusion and informative sensors](Neural_PDE_Exercise_05_inverse_problem.ipynb) | a sensor-location comparison across seeds and an identifiability argument |
| Checking a neural solution | [06 · Build an independent solution audit](Neural_PDE_Exercise_06_solution_checks.ipynb) | an audit table and a refined finite-difference reference |

Allow about **35–50 minutes per exercise**, with more time for the optional
extensions. The notebooks are independent; their order follows the slides.
The [seven laboratory notebooks](../NOTEBOOK_README.md) provide longer worked
experiments, including higher-dimensional, time-dependent, and parametric PDEs.

## Run a notebook

Use Python 3.11 or newer. From `3. Neural networks for solving PDEs/`:

```sh
python -m pip install -r experiments/requirements-notebook.txt
python -m jupyter lab exercises/Neural_PDE_Exercise_01_data_to_pde.ipynb
```

Select **Restart Kernel and Run All Cells**. Each notebook includes its own imports,
model definitions, data, and checks; no laboratory output or earlier notebook is
required. The exercises use NumPy, Matplotlib, and PyTorch from the existing
laboratory environment. Computation uses CPU / float64, a fixed seed, and short
training budgets; the baseline runs typically take seconds to under a minute,
depending on the machine.

## Working on the exercises

1. Read the equations and write down a prediction before running the baseline.
2. Edit the settings marked `EDIT HERE`, changing one factor at a time.
3. Rerun all cells and record the seed, budget, settings, and independent diagnostics.
4. Keep the requested table or plot and answer the reflection questions.

The checks verify structural properties and finite computations. They do not
assume that one optimizer or formulation wins for every seed. Training losses,
solution errors, boundary errors, and residuals are reported separately. Exact
solutions support validation; inverse-problem observations are identified
explicitly. Exercise 05 uses a small sine-feature state to isolate coefficient
identifiability before an optional extension to a general network.

These exercise notebooks are edited directly. They keep figures inside the
notebook and do not write into `notebook_outputs/`. The existing laboratory
builder continues to manage only the original seven laboratories and their index.

**Back to:** [chapter overview](../README.md) · [laboratory guide](../NOTEBOOK_README.md)

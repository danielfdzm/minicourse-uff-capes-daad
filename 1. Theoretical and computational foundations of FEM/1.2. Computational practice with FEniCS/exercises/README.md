# FEniCSx practical exercises

Two self-contained notebooks follow the examples from the two decks. Each gives
a runnable baseline, asks for a controlled numerical experiment, and identifies
the checks and interpretation to include in your answer.

| Deck / example | Notebook | Practical task |
|---|---|---|
| Part I · setup and mathematical background | [Part I](FEniCSx_Part_I_Exercises.ipynb) | check the environment and map each weak-form object to its implementation |
| 1 · Poisson | [Part I](FEniCSx_Part_I_Exercises.ipynb) | refine the mesh and compare true integrated errors for P1 and P2 |
| 2 · Neumann refinement | [Part I](FEniCSx_Part_I_Exercises.ipynb) | compare local refinement with uniform refinement for a screened problem |
| 3 · L-shaped eigenfunction | [Part I](FEniCSx_Part_I_Exercises.ipynb) | refine the domain mesh and check the generalized eigenpair residual |
| 4 · Heat equation | [Part I](FEniCSx_Part_I_Exercises.ipynb) | compare time integrators, energy decay and independent time refinement |
| 5 · Nonlinear Poisson | [Part I](FEniCSx_Part_I_Exercises.ipynb) | change diffusion strength and distinguish nonlinear convergence from PDE accuracy |
| 6 · Stokes | [Part I](FEniCSx_Part_I_Exercises.ipynb) | inspect a Taylor–Hood cavity solve, pressure normalization and incompressibility |
| 7 · 3D p-Laplacian | [Part II](FEniCSx_Part_II_Exercises.ipynb) | vary the exponent and regularisation in a nonlinear variational problem |
| 8 · Nonlocal parabolic PDE | [Part II](FEniCSx_Part_II_Exercises.ipynb) | track an integral-dependent diffusion coefficient and refine time |
| 9 · Degenerate nonlocal null control | [Part II](FEniCSx_Part_II_Exercises.ipynb) | inspect control placement, terminal error and the effect of degeneracy |
| 10 · Optimal control with drift | [Part II](FEniCSx_Part_II_Exercises.ipynb) | compare terminal accuracy and control cost, and check a discrete adjoint gradient |
| 11 · Coupled reaction–diffusion | [Part II](FEniCSx_Part_II_Exercises.ipynb) | vary reaction and diffusion parameters and check time-step sensitivity |

## Run a practical

Follow the [environment and kernel setup](../NOTEBOOK_README.md). From
`1.2. Computational practice with FEniCS/`, with `fenicsx-env` active:

```sh
python -m jupyter lab exercises/FEniCSx_Part_I_Exercises.ipynb
# Or start with the advanced examples:
python -m jupyter lab exercises/FEniCSx_Part_II_Exercises.ipynb
```

Choose the **Python (FEniCSx)** kernel and select **Restart Kernel and Run All Cells**.
The notebooks use DOLFINx 0.10.x, real float64 PETSc, and a single process. Each
contains its own setup and starter solvers; earlier notebooks are not required.

## Numerical models and deliverables

Part I uses FEniCSx for the finite element experiments. In Part II, Examples 7
and 8 use FEniCSx for p-Laplacian and nonlocal diffusion problems. Example 10
assembles FEM matrices in FEniCSx and advances them in SciPy to expose the
discrete adjoint. Examples 9 and 11 use explicitly labelled NumPy/SciPy
finite-difference models for degenerate control and reaction–diffusion.
These are teaching models for the corresponding slide topics;
they do not reproduce every detail of the larger slide experiments. The notebook
states the assumptions and limitations before each task.

Work through each deck over several sessions. For every example, record your
settings and prediction, run a comparison, and provide a table or plot with an
explanation. Check the requested residuals, errors or conservation properties;
a solver returning successfully is only the first check. Complete the notebook's
submission checklist after your comparisons. Saved baseline outputs are examples
of the diagnostics, not completed answers to the practical tasks.

The notebooks keep their figures inside the file and are edited directly. The
original [slide scripts](../README.md#examples) and their outputs keep their
existing locations.

**Back to:** [chapter overview](../README.md) · [laboratory guide](../NOTEBOOK_README.md)

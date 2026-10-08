# FEniCSx laboratory

The [Poisson and heat laboratory](FEniCSx_01_poisson_and_heat.ipynb) turns the
weak forms into DOLFINx code, checks P1/P2 convergence against an analytical
solution, and compares backward Euler with Crank–Nicolson. It is self-contained
and includes saved static plots for GitHub previews.

| Notebook | Experiments and tasks |
|---|---|
| [01 · Poisson convergence and heat flow](FEniCSx_01_poisson_and_heat.ipynb) | mesh, function space, boundary constraints, variational solve, integrated errors and time stepping |
| [Part I practical](exercises/FEniCSx_Part_I_Exercises.ipynb) | setup and the math-to-code dictionary, then Examples 1–6: Poisson, Neumann refinement, eigenfunctions, heat, nonlinear diffusion and Stokes |
| [Part II practical](exercises/FEniCSx_Part_II_Exercises.ipynb) | Examples 7–11: p-Laplacian, nonlocal diffusion, control and reaction–diffusion |

The [exercise guide](exercises/README.md) maps each slide example to a task and
describes the numerical models used in the advanced practical.

## Run a notebook

From `1.2. Computational practice with FEniCS/`:

```sh
conda env create -f environment.yml
conda activate fenicsx-env
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name fenicsx-env --display-name "Python (FEniCSx)"
python -m jupyter lab FEniCSx_01_poisson_and_heat.ipynb
```

If the environment already exists, activate it and install the notebook tools.
Choose **Python (FEniCSx)** in Jupyter, then **Restart Kernel and Run All Cells**.
The notebooks target DOLFINx **0.10.x**, real-valued float64 PETSc, and a normal
serial kernel. Their setup checks report a mismatched version or scalar type.
The legacy `fenics` package cannot run these notebooks. Part I's Stokes solve
uses PETSc's MUMPS factorisation for the saddle-point system; use a PETSc build
with MUMPS support, as in the tested conda-forge environment.

Each notebook contains its own imports, forms, solver helpers and data. Plotting
uses Matplotlib, so the notebooks do not need a PyVista display or the existing
saved results. First execution can take longer while FEniCSx compiles the forms.

## Outputs and editing

Saved plots and small numerical tables stay inside the notebooks. The notebooks
do not overwrite `experiments/figures/` or `experiments/results/`; those folders
remain the outputs of the original slide scripts. Edit the notebooks directly.

Keep PDE errors separate from linear or nonlinear solver convergence. Use an
analytical expression when checking the Poisson solution, and refine space and
time independently in a heat experiment. The practicals ask for these checks
alongside the plots.

**Back to:** [chapter overview](README.md) · [practical exercises](exercises/README.md)

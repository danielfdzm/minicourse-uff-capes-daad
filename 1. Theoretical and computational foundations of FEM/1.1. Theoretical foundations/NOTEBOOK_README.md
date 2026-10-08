# FEM foundations laboratory

Two self-contained notebooks turn the weak formulations and error estimates into
small numerical experiments. Both implement one-dimensional P1 finite elements
in NumPy/SciPy, so the matrix entries, quadrature and boundary constraints remain
visible. FEniCSx implementations follow in [Chapter 1.2](../1.2.%20Computational%20practice%20with%20FEniCS/NOTEBOOK_README.md).

| Notebook | Experiments |
|---|---|
| [01 · Assembly and convergence](FEM_01_assembly_and_convergence.ipynb) | stiffness and mass matrices, Galerkin orthogonality, energy minimisation, true L² and H¹ errors, nonzero boundary data |
| [02 · Adaptivity and heat flow](FEM_02_adaptivity_and_heat.ipynb) | residual indicators, marking and refinement, error versus degrees of freedom, time-step stability and temporal convergence |
| [Practical exercises](exercises/FEM_Foundations_Exercises.ipynb) | nine tasks covering every section of the theory deck; see the [exercise guide](exercises/README.md) |

## Run a notebook

Use Python 3.11 or newer. From `1.1. Theoretical foundations/`:

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyter lab FEM_01_assembly_and_convergence.ipynb
```

Select **Restart Kernel and Run All Cells**. Each notebook contains its own
imports, assembly routines, data and plotting setup. Computation uses CPU /
float64 and a fixed random seed. No slide figures or other notebook outputs are
needed, and the baseline calculations are small enough for a laptop.

## Read the results

Saved static plots and numerical checks can be read on GitHub. Figures remain
inside the notebooks; rerunning them does not replace the slide figures. These
notebooks are edited directly and do not need a generator.

The assembly notebook distinguishes algebraic residuals from discretisation
errors. The adaptivity notebook compares meshes at similar numbers of unknowns,
and separates temporal error from spatial error using a semidiscrete reference.
The practical asks you to make these distinctions in your own experiments.

**Back to:** [chapter overview](README.md) · [practical exercises](exercises/README.md)

# FEM foundations practical exercises

The [practical notebook](FEM_Foundations_Exercises.ipynb) follows all nine sections
of Chapter 1.1. It includes a small P1 assembly routine and runnable baselines;
your task is to extend the experiments and explain the numerical evidence.

| Course section | Practical task |
|---|---|
| Motivation and roadmap | derive a manufactured forcing and find the resolution needed for several frequencies |
| Weak formulations and well-posedness | vary the diffusion coefficient, examine conditioning and discuss Neumann compatibility |
| Galerkin approximation | test orthogonality and compare energies with discrete perturbations |
| Meshes and finite element spaces | compare graded meshes, plot hat functions and map reference elements |
| Assembly and discrete systems | assemble by hand and diagnose an incorrect boundary lift |
| Error estimates and convergence | measure integrated errors for smooth and reduced-regularity states |
| A posteriori estimation and adaptivity | implement a marking loop and compare errors at similar numbers of unknowns |
| Evolution PDEs: space and time | compare time integrators using the consistent-mass stability limit |
| Checking the solution | build a verification report and test it against a deliberate bug |

## Run the practical

From `1.1. Theoretical foundations/`, using Python 3.11+:

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyter lab exercises/FEM_Foundations_Exercises.ipynb
```

Select **Restart Kernel and Run All Cells** to reproduce the baseline. The
notebook runs independently of the [worked laboratories](../NOTEBOOK_README.md)
and uses NumPy/SciPy, CPU / float64, with no FEniCS installation required.

Allow two sessions of 60–90 minutes. For each task, write a prediction, edit the
marked cell, rerun the experiment, and record a table or plot with your explanation.
The supplied assertions check the baseline; a completed practical also needs the
requested comparisons and the final verification report. Figures are embedded in
the notebook and do not change the slide assets.

**Back to:** [theory overview](../README.md) · [laboratory guide](../NOTEBOOK_README.md)

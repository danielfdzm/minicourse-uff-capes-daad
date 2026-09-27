# 1.2 Computational practice with FEniCS

These two decks show how to implement the weak formulations from Chapter 1.1 in
FEniCSx. The accompanying scripts generate the results shown in the slides.

| Deck | Content |
|---|---|
| **[Part I: from weak forms to working code (PDF, 38 slides)](1.2_fenics_part1.pdf)** | installation, the math-to-code dictionary, Examples 1–6, and backup slides with complete legacy FEniCS and FEniCSx Poisson codes |
| **[Part II: advanced PDE examples (PDF, 35 slides)](1.2_fenics_part2.pdf)** | writing nonlinear residuals and solving Examples 7–11 |

## Examples

| Example on the slides | Script in `experiments/` | Main figures in `experiments/figures/` |
|---|---|---|
| 1 · Poisson equation with a known solution | `01_poisson.py`, `02_convergence.py` | `poisson_solution*.png`, `convergence_plot_p1.png`, `convergence_plot_p2.png` |
| 1 · the slide code, line by line | `01_poisson_slide_code_fenicsx.py`, `01_poisson_slide_code_legacy.py` | `poisson_slide_code_fenicsx.png` |
| 2 · Neumann problem with local mesh refinement | `08_neumann_refinement.py` | `neumann_refinement.png`, `neumann_refinement_error_metric.png` |
| 3 · L-shaped Laplacian eigenfunction | `07_lshape_eigenfunction.py` | `lshape_eigenfunction_2d.png`, `lshape_eigenfunction_3d_matlab.png` |
| 4 · Heat equation | `03_heat_equation.py` | `heat_equation.png`, `heat_decay_metrics.png` |
| 5 · Nonlinear Poisson equation | `04_nonlinear.py` | `nonlinear_poisson*.png`, `nonlinear_kappa_hist.png` |
| 6 · Stokes flow in a lid-driven cavity | `05_stokes.py` | `stokes_cavity.png`, `stokes_cavity_3d.png` |
| 7 · 3D p-Laplacian | `10_plaplacian_3d.py` | `p_laplacian_3d.png` |
| 8 · Nonlocal parabolic PDE | `11_nonlocal_variants.py` (single case: `09_nonlocal_parabolic.py`) | `nonlocal_variants.png`, `nonlocal_parabolic_solution.png` |
| 9 · Degenerate nonlocal null control | `14_degenerate_nonlocal_control.py` | `degenerate_nonlocal_control*.png` |
| 10 · Optimal control of a heat equation with drift | `12_heat_drift_control.py` | `heat_drift_control.png` |
| 11 · Coupled reaction–diffusion system | `13_reaction_diffusion_system.py` | `reaction_diffusion_system.png` |
| extra · from FEniCSx matrices to SciPy | `06_matrix_extraction.py` | `eigenfunctions.png`, `eigenmode1_3d.png`, `eigenvalue_*.png` |

Scripts 13 and 14 are compact NumPy/SciPy finite-difference solvers that produce the slide
figures; the Part II slides show the finite element formulation you would write in FEniCSx.
Solution fields (XDMF/HDF5, for ParaView) and CSV tables are written to
`experiments/results/`. Console output from earlier runs is kept in `results/logs/`.

## Run the examples

```sh
conda env create -f environment.yml     # DOLFINx 0.10, PyVista, gmsh, SciPy, Matplotlib
conda activate fenicsx-env
python experiments/01_poisson.py
python experiments/14_degenerate_nonlocal_control.py --mesh feature --nx 60 --nt 120 --refine-iters 50
```

- A script can be started from any directory: outputs always go to `experiments/figures/`
  and `experiments/results/`, overwriting the previous versions. The slides pick up the new
  figures on their next build.
- On a machine without a display (SSH, CI), run `export PYVISTA_OFF_SCREEN=true` first so
  that PyVista can take its screenshots.
- `01_poisson_slide_code_legacy.py` is the legacy-FEniCS version of the slide code
  (`from fenics import *`) and needs FEniCS 2019 rather than DOLFINx.
- Script 14 writes figures to `--outdir` and tables to `--datadir`; `--help` lists its
  solver options.

## Build the slides

```sh
latexmk -pdf 1.2_fenics_part1.tex
latexmk -pdf 1.2_fenics_part2.tex
```

The slides read their figures from `experiments/figures/`.

**Previous:** [1.1 Theoretical foundations](../1.1.%20Theoretical%20foundations)

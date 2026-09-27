# Figure sources

This folder contains the figures used by `../1.1_fem_foundations.tex`.
The scripts named below (`code/` and `tools/`) belong to the authoring
repository and are **not included here**. You can build the slides with the
supplied images; the commands below document how they were produced.

The reference-run figures were generated with:

```sh
PYTHONPATH=code python -m l1.fem1d_numpy --output-dir results/reference_runs/fem_numpy
python tools/run_reference.py --steps 2000 --seeds 0 1 2
```

In that repository, `tools/refresh_tables_figures.py` copies plots from
`results/reference_runs` and updates the LaTeX tables without retraining.
No FEniCSx solution figure was generated there because DOLFINx was unavailable.
The TikZ diagrams in the slides illustrate the method separately from the
numerical results.

`python tools/make_teaching_figures.py` produces the coarse Benchmark A
figure by solving the eight-element NumPy problem.

## Mesh, basis, and convergence figures

The following command produces the thirteen figures listed below:

```sh
PYTHONPATH=code python tools/make_advanced_figures.py
```

It also writes five LaTeX tables to `notes/generated/` and records their
values in `results/advanced_chapter1_numbers.json`. The shape functions use
`l1.lagrange1d`; the meshes use `l1.fem2d_numpy` and `l1.fem3d_numpy`. The
convergence panels show errors measured in refinement studies.

| File | What it shows |
|---|---|
| `lagrange_shape_functions.png` | Reference P1, P2 and P3 bases on `[0,1]` |
| `lagrange_convergence.png` | Measured 1D rates, and error against unknown count |
| `mesh2d_gallery.png` | Four triangulations coloured by shape regularity |
| `p1_p2_dof_layout.png` | P1 against P2 degrees of freedom; two P2 shape surfaces |
| `poisson2d_solution.png` | Benchmark C: coarse surface, fine solution, signed error |
| `sparsity_pattern.png` | P1 and P2 sparsity, and nonzeros per row under refinement |
| `mesh3d_cube.png` | The six Kuhn tetrahedra, the assembled mesh, a cutaway |
| `poisson3d_slices.png` | The 3D solution on a cut, a mid-plane slice, measured rates |
| `heat1d_theta.png` | Benchmark D: space-time carpet, snapshots, temporal order |
| `heat_stability.png` | Forward Euler either side of the critical step; the `h^2` law |
| `heat2d_snapshots.png` | Benchmark E under the theta-scheme |
| `isoparametric_geometry.png` | Straight against curved cells on a disc |
| `lshape_singularity.png` | Benchmark F: the corner singularity and its reduced rate |

Each figure has two versions. The plain filename includes a caption for the
notes; the `_slide` version omits it to leave more room for the plots. Both
use the same computed data.

The palette in `tools/figure_style.py` uses three colours for categories, a
sequential scale for magnitudes, and a diverging scale for signed quantities.
Series also have direct labels so readers can identify them without relying
on colour alone.

## Reference-run plots not used by the slides

`reference_runs/` holds twelve plots from the Benchmark A reference runs: the
NumPy FEM error curves (`fem_L2_error.png`, `fem_H1_semi_error.png`) and the
comparison of FEM with a supervised network, a PINN and Deep Ritz
(`supervised_*`, `data_splits`, `training_validation_loss`, `pinn_*`,
`deep_ritz_*`, `comparison_*`). The current slides do not include them; they are
kept for reference.

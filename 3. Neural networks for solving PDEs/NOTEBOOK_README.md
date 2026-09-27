# Neural PDE Laboratory

The 13 experiments are split into **seven self-contained notebooks**. Each includes
its own imports, plotting helpers, and saved static plots; you can run any notebook
in a fresh kernel without first running another one.

| Notebook | Experiments |
|---|---|
| [01 · Membranes, PINNs, Deep Ritz and boundary conditions](Neural_PDE_01_foundations.ipynb) | 1–3 |
| [02 · Poisson surfaces and limits of continuous losses](Neural_PDE_02_poisson_and_limits.ipynb) | 4–5 |
| [03 · Inverse diffusion and sensor placement](Neural_PDE_03_inverse.ipynb) | 6 |
| [04 · High dimensions and the annulus](Neural_PDE_04_dimensions_and_geometry.ipynb) | 7–8 |
| [05 · Reaction–diffusion patterns](Neural_PDE_05_reaction_diffusion.ipynb) | 9 |
| [06 · Fourier features and adaptive collocation](Neural_PDE_06_features_and_sampling.ipynb) | 10–11 |
| [07 · Heat flow and parameter-conditioned PINNs](Neural_PDE_07_time_and_parameters.ipynb) | 12–13 |

[Neural_PDE_Laboratory.ipynb](Neural_PDE_Laboratory.ipynb) is now a small navigation
notebook, so existing links still work. The individual notebooks omit embedded
JavaScript and animation data. Their saved figures and diagnostics can be browsed
on GitHub; interactive explorers remain available as standalone offline HTML files.
Download `notebook_outputs/` and open [index.html](notebook_outputs/index.html) in a browser.

## Run a notebook

Use Python 3.11 or newer. From this folder (`3. Neural networks for solving PDEs/`):

```sh
python -m pip install -r experiments/requirements-notebook.txt
python -m jupyter lab Neural_PDE_01_foundations.ipynb
```

Choose another filename from the table to start with a different topic. Select
**Restart Kernel and Run All Cells**. Each notebook computes its own data and can be
copied and run independently of the slide sources, experiment scripts, and other
notebooks once the dependencies are installed.

The initial cells contain the seed and training budgets. Computations use CPU/float64
and short training runs; `RUN_MODE = "thorough"` increases budgets where supported.
Runtime and numerical diagnostics are recorded separately for each notebook.

## Outputs

All notebooks write to `notebook_outputs/` in the kernel's working directory:

- Static figure sets as PNG and vector PDF, with saved PNG previews in the notebooks.
- Seven self-contained interactive HTML explorers across the collection: membrane
  motion, Poisson fields, the annulus, reaction–diffusion, modal learning, heat flow,
  and the parameterized PDE family.
- `index_<part>.html`, a gallery for the notebook that was run.
- `run_summary_<part>.json`, with versions, seeds, settings and measured errors.
- `computed_fields_<part>.npz`, with that notebook's numerical fields and diagnostics.

For example, notebook 01 writes `run_summary_01_foundations.json`. Unique report
names let you run the notebooks in any order. The supplied `index.html` links the
whole collection; the original `run_summary.json` and `computed_fields.npz` retain
the baseline results from the earlier combined notebook.

GitHub shows the saved static plots. For rotation, sliders and animations, open the
exported HTML files locally in a browser. They contain their own JavaScript and work
offline. Read the axes: height can represent a field value, and some plots use time
or a PDE parameter as an axis. Exact solutions check errors after forward training.

## Rebuild the collection

Edit source cells in `experiments/notebook_parts/`. The builder inserts the shared
setup into each notebook, assigns experiments to parts, and generates the index.
To rebuild and execute all seven in separate fresh kernels:

```sh
python experiments/build_notebook.py --execute
```

To rebuild and execute just one part:

```sh
python experiments/build_notebook.py --part 03_inverse --execute
```

Omitting `--execute` writes unexecuted notebooks. The builder enforces a 5 MB size
budget per notebook to keep previews manageable; interactive data stays in HTML.

# 3. Neural networks for solving PDEs

**[Slides: Neural networks for solving PDEs (PDF, 83 slides)](3_neural_networks_for_pdes.pdf)**
&nbsp;·&nbsp; source [`3_neural_networks_for_pdes.tex`](3_neural_networks_for_pdes.tex)

Use a neural network to solve a PDE, then check whether the solution is accurate.
The chapter covers PINNs, Deep Ritz, weak formulations, and inverse problems:

| Section | What it covers |
|---|---|
| From data to PDE solvers | data-driven learning, and how a PDE enters the loss |
| Strong-form PINNs | collocation, quadrature, stability and error, training and approximation limits |
| Network design and boundary constraints | regularity and architecture, soft penalties versus exact (hard) boundary conditions |
| Variational and weak methods | Deep Ritz energies and weak formulations |
| Hybrid PDE–data methods | hybrid residual losses, inverse problems and data assimilation, limitations |
| Checking a neural solution | choosing a formulation and checking errors |

## What else is in this folder

| Item | Purpose |
|---|---|
| [`experiments/`](experiments) | the scripts, saved results and figures behind the numbers on the slides; see [`experiments/README.md`](experiments/README.md) for configurations, measured results and how to reproduce them |
| [`Neural_PDE_Laboratory.ipynb`](Neural_PDE_Laboratory.ipynb) | index of seven independent notebooks covering 13 experiments, with saved plots; see [`NOTEBOOK_README.md`](NOTEBOOK_README.md) |
| [`notebook_outputs/`](notebook_outputs) | the laboratory's figures (PNG and PDF) and seven interactive HTML explorers; open `index.html` in a browser |
| `Neural_PDE_Laboratory_bundle.zip` | all seven notebooks, the index, sources and outputs in one download |

## Laboratory notebooks

| Notebook | Experiments |
|---|---|
| [01 · Membranes, PINNs, Deep Ritz and boundary conditions](Neural_PDE_01_foundations.ipynb) | 1–3 |
| [02 · Poisson surfaces and limits of continuous losses](Neural_PDE_02_poisson_and_limits.ipynb) | 4–5 |
| [03 · Inverse diffusion and sensor placement](Neural_PDE_03_inverse.ipynb) | 6 |
| [04 · High dimensions and the annulus](Neural_PDE_04_dimensions_and_geometry.ipynb) | 7–8 |
| [05 · Reaction–diffusion patterns](Neural_PDE_05_reaction_diffusion.ipynb) | 9 |
| [06 · Fourier features and adaptive collocation](Neural_PDE_06_features_and_sampling.ipynb) | 10–11 |
| [07 · Heat flow and parameter-conditioned PINNs](Neural_PDE_07_time_and_parameters.ipynb) | 12–13 |

## Quick start

```sh
python -m pip install -r experiments/requirements-notebook.txt
python -m jupyter lab Neural_PDE_01_foundations.ipynb # first notebook
latexmk -pdf 3_neural_networks_for_pdes.tex            # rebuild the slides
```

For training commands and figure generation, see
[`experiments/README.md`](experiments/README.md). Use the footer IDs (`L3-S001`, …)
to refer to individual slides.

**Previous chapter:** [2. Fundamentals of Machine Learning](../2.%20Fundamentals%20of%20Machine%20Learning)

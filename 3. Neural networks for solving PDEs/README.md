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
| [`Neural_PDE_Laboratory.ipynb`](Neural_PDE_Laboratory.ipynb) | a notebook with 13 experiments, explanations, and saved outputs; see [`NOTEBOOK_README.md`](NOTEBOOK_README.md) |
| [`notebook_outputs/`](notebook_outputs) | the notebook's figures (PNG and PDF) and seven interactive HTML explorers; open `index.html` in a browser |
| `Neural_PDE_Laboratory_bundle.zip` | the notebook, its sources and outputs in a single download |

## Quick start

```sh
python -m pip install -r experiments/requirements-notebook.txt
python -m jupyter lab Neural_PDE_Laboratory.ipynb     # the laboratory
latexmk -pdf 3_neural_networks_for_pdes.tex            # rebuild the slides
```

For training commands and figure generation, see
[`experiments/README.md`](experiments/README.md). Use the footer IDs (`L3-S001`, …)
to refer to individual slides.

**Previous chapter:** [2. Fundamentals of Machine Learning](../2.%20Fundamentals%20of%20Machine%20Learning)

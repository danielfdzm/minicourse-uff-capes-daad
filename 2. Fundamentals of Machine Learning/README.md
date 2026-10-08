# 2. Fundamentals of Machine Learning

**[Slides: Neural networks for supervised regression (PDF, 80 slides)](2_machine_learning_fundamentals.pdf)**
&nbsp;·&nbsp; source [`2_machine_learning_fundamentals.tex`](2_machine_learning_fundamentals.tex)

Learn the machine learning needed for Chapter 3 by fitting one dataset: noisy
samples of a hidden function. We start with a straight line in NumPy, then train
neural networks in PyTorch and check how well they predict unseen data.

| Slides | What they cover |
|---|---|
| 1–14 | the supervised learning task, regression versus interpolation, population and empirical risk, a toy dataset, linear regression and the normal equations |
| 15–35 | networks as nested functions, neurons and layers, universal approximation (Cybenko, Pinkus/LLPS) with proof sketches, and what the theorems do not say |
| 36–45 | shapes and parameter counts, activations, forward passes, losses and gradients in PyTorch |
| 46–60 | gradient descent and learning rates, non-convex objectives, stochastic and projected gradient descent, momentum, RMSProp and Adam, backpropagation, the training loop |
| 61–71 | L2 and L1 regularisation, elastic net, early stopping, width and depth, underfitting and overfitting, learning curves and checkpointing |
| 72–80 | test evaluation and leakage, metrics, repeated seeds, hyperparameter search, common bugs, a formula-to-code dictionary, experiment records |

Use the footer IDs (`L2-S01` … `L2-S80`) to refer to individual slides.

## Notebooks

The [Machine Learning Laboratory](NOTEBOOK_README.md) contains three
self-contained worked notebooks. Each generates its own data and includes saved
static plots that can be read on GitHub.

| Notebook | What to explore |
|---|---|
| [01 · Regression and risk](ML_01_regression_and_risk.ipynb) | least squares, noisy interpolation, population risk and repeated datasets |
| [02 · Networks and optimisation](ML_02_networks_and_optimisation.ipynb) | constructive ReLU approximation, shapes, autograd, learning rates and optimiser comparisons |
| [03 · Regularisation and selection](ML_03_regularisation_and_selection.ipynb) | L1/L2/elastic net, ridge penalties, early stopping, repeated seeds and final test metrics |

Separate [practical exercise notebooks](exercises/README.md) cover all six slide
ranges above with editable starter code, hints, feedback and interpretation tasks.
They run independently of the worked examples.

Use Python 3.11 or newer. From this folder:

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyter lab ML_01_regression_and_risk.ipynb
```

Select **Restart Kernel and Run All Cells**. See the
[notebook guide](NOTEBOOK_README.md) for data, runtime and evaluation conventions.

## Build

```sh
latexmk -pdf 2_machine_learning_fundamentals.tex
```

The deck reads its figures from [`figures/`](figures).

**Previous chapter:** [1. Foundations of FEM](../1.%20Theoretical%20and%20computational%20foundations%20of%20FEM)
&nbsp;·&nbsp; **Next chapter:** [3. Neural networks for solving PDEs](../3.%20Neural%20networks%20for%20solving%20PDEs)

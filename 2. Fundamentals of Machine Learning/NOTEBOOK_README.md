# Machine Learning Laboratory

Three **self-contained notebooks** turn the Chapter 2 regression example into
computational experiments. Each includes its own data, imports, plotting helpers
and saved static plots; run any notebook in a fresh kernel without first running
another one.

| Notebook | Experiments | Slides |
|---|---|---|
| [01 · Regression, interpolation and population risk](ML_01_regression_and_risk.ipynb) | affine and cubic least squares; noisy interpolation; risk decomposition; repeated datasets and learning curves | L2-S01–L2-S14 |
| [02 · Networks, automatic differentiation and optimisation](ML_02_networks_and_optimisation.ipynb) | constructive ReLU approximation; tensor shapes and gradient checks; quadratic learning-rate stability; momentum SGD and Adam | L2-S15–L2-S60 |
| [03 · Regularisation, early stopping and model selection](ML_03_regularisation_and_selection.ipynb) | L1, L2 and elastic-net shrinkage; ridge selection; width/depth and repeated seeds; restored checkpoints and final test metrics | L2-S61–L2-S80 |

The [practical exercises](exercises/README.md) provide separate learner notebooks
with editable starter cells, hints, feedback and written interpretation tasks.
Their coverage follows all six slide ranges in the [chapter guide](README.md).

## Run a notebook

Use Python 3.11 or newer. From this folder (`2. Fundamentals of Machine Learning/`):

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyter lab ML_01_regression_and_risk.ipynb
```

Choose another filename from the table to start with a different topic. Select
**Restart Kernel and Run All Cells**. No dataset downloads, slide assets or
earlier notebook results are needed. The notebooks also work when copied
individually once the dependencies are installed.

The initial cell contains the seed and `RUN_MODE`. Computations use CPU/float64,
one PyTorch thread and short training runs. `RUN_MODE = "thorough"` doubles
repetition or training budgets where applicable. The default signal is
`sin(pi*x)`, matching the slides; the regularisation notebook deliberately uses
fewer observations and more noise to expose differences between model choices.

## Read the results

Static PNG plots and text diagnostics are saved inside each notebook for GitHub
previews. Running a notebook refreshes those outputs; it does not create external
figures, model checkpoints or result folders. Version information appears in the
setup cell and each worked notebook prints its runtime.

The first notebook uses known signal values to diagnose error in a synthetic
experiment. The third notebook selects penalties and architectures using
validation data, then evaluates its two predeclared model families on independent
test data. Its final cell prints an experiment record with settings, checkpoint
epochs, versions and metrics. The repeated neural seeds measure initialisation
variability on a fixed split; they do not estimate variability across datasets.

Each practical runs with its starter implementations. Feedback such as “revise the
manual gradient” identifies a learner task still to complete; it is not an execution
failure. Practical 03 keeps the test set closed by default until the learner records
and freezes their validation-based choices.

**Previous chapter:** [1. Foundations of FEM](../1.%20Theoretical%20and%20computational%20foundations%20of%20FEM)
&nbsp;·&nbsp; **Next chapter:** [3. Neural PDE Laboratory](../3.%20Neural%20networks%20for%20solving%20PDEs/NOTEBOOK_README.md)

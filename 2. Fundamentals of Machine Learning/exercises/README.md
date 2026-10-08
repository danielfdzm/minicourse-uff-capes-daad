# Machine Learning Practical Exercises

Three **self-contained exercise notebooks** cover every section of Chapter 2.
Each starts with a runnable baseline, then asks you to edit code, compare numerical
results and explain what the results establish. Hints and feedback support the
tasks without replacing the written interpretation.

| Notebook | Practical work | Suggested time |
|---|---|---|
| [01 · Regression, risk and approximation](ML_Exercises_01_regression_and_approximation.ipynb) | derive a population risk; compare regression models; construct a ReLU interpolant; interpret approximation theorems | 60–90 min |
| [02 · Shapes, automatic differentiation and optimisation](ML_Exercises_02_autograd_and_optimisation.ipynb) | check parameters and shapes; implement a gradient; study learning rates, projection and mini-batches; compare neural training runs | 75–105 min |
| [03 · Regularisation, checkpointing and honest evaluation](ML_Exercises_03_regularisation_and_evaluation.ipynb) | implement shrinkage; restore a checkpoint; select across repeated seeds; release a final test; record metrics and leakage checks | 75–105 min |

## Coverage

| Slides | Section topics | Exercise tasks |
|---|---|---|
| L2-S01–L2-S14 | supervised regression; interpolation versus learning; empirical and population risk; affine least squares | 01 · A–B |
| L2-S15–L2-S35 | neurons and layers; network representation; universal approximation and its limitations | 01 · C |
| L2-S36–L2-S45 | tensor shapes; parameter counts; activations; forward passes; losses and gradients | 02 · A–B |
| L2-S46–L2-S60 | learning rates; non-convexity; stochastic and projected gradients; momentum, RMSProp and Adam; backpropagation and training loops | 02 · B–D |
| L2-S61–L2-S71 | L1, L2 and elastic net; early stopping; width/depth; underfitting/overfitting; learning curves and checkpointing | 03 · A–C |
| L2-S72–L2-S80 | test evaluation; metrics; repeated seeds; hyperparameter search; implementation bugs; formula-to-code checks and experiment records | 03 · C–D |

## Run an exercise

Use Python 3.11 or newer. From the chapter folder:

```sh
python -m pip install -r requirements-notebook.txt
python -m jupyter lab exercises/ML_Exercises_01_regression_and_approximation.ipynb
```

Select **Restart Kernel and Run All Cells** once to inspect the starter results.
Work through the numbered tasks, edit cells marked `EDIT`, and write your
observations in the answer cells. Restart and run again after making changes to
check that the notebook still works without hidden state.

Saved outputs show the supplied baselines, including feedback for unfinished
tasks. There are no intentionally crashing placeholders. Assertions check tensor
shapes and numerical consistency; learner feedback reports whether the requested
implementation is complete. Each notebook contains its own data and setup.

For each practical, submit the edited notebook with the requested comparison
table or plots and the written interpretation. In practical 03, complete and
record model selection before setting `RELEASE_TEST = True`; test performance
must not determine the settings you report.

**Worked examples:** [Machine Learning Laboratory](../NOTEBOOK_README.md)
&nbsp;·&nbsp; **Slides:** [Chapter 2 guide](../README.md)

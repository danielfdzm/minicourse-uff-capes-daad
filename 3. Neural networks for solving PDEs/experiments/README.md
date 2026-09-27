# Python experiments for the PDE slides

To work through the experiments interactively, open
[Neural PDE Laboratory](../NOTEBOOK_README.md). Its seven independent notebooks include the main slide
examples, saved outputs, and further experiments on membranes, sensor placement,
sampling, curved domains, and time-dependent PDEs. See the
[notebook guide](../NOTEBOOK_README.md) for the full list and setup instructions.

These scripts generate the numerical results, source excerpts, and figures in
`3_neural_networks_for_pdes.pdf`. Training runs locally with PyTorch on one
CPU thread in `float64`. The supplied results were produced with Python 3.14.0,
NumPy 2.3.5, PyTorch 2.10.0, and Matplotlib 3.10.7; package versions are pinned
in [requirements.txt](requirements.txt). The equations and figure labels follow the
[shared notation guide](notation.md).

Run the following commands from the chapter folder
(`3. Neural networks for solving PDEs/`, one level above this file). A virtual environment keeps
the dependencies separate from other projects. Compiling the slides also
requires a LaTeX installation with `latexmk` and the packages used by the deck.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r experiments/requirements.txt
python3 experiments/forward_1d.py
python3 experiments/poisson_2d.py
python3 experiments/inverse_1d.py
python3 experiments/classical_accuracy.py
python3 experiments/make_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error 3_neural_networks_for_pdes.tex
```

The three neural training scripts took approximately 19, 19, and 13 seconds,
respectively, on the machine used for the supplied results. Runtime and the
last numerical digits depend on the machine and software environment. The
forward experiment was run twice in that environment, and all 45 saved arrays
were bitwise identical. To regenerate only figures and excerpts from the
supplied numerical results, run `make_figures.py` followed by `latexmk`.

The five experiments use the following problems and configurations:

| Experiment | Training setup | Evaluation setup |
| --- | --- | --- |
| 1D forward PINN | Solve `-u'' = pi² sin(pi x)` on `[0,1]`. Seed 7; `1–32–32–1` tanh network; `u=x(1-x)net(x)`; 128 Gauss–Legendre nodes. | 1,001 equally spaced points, distinct from the quadrature nodes; exact solution `sin(pi x)`. |
| 2D forward PINN | Solve `-Delta u = 2 pi² sin(pi x)sin(pi y)` on the unit square. Seed 7; `2–32–32–32–1` tanh network; hard factor `x(1-x)y(1-y)`; 1,024 fixed scrambled Sobol points. Network inputs are scaled to `[-1,1]²`. | Independent `101×101` uniform grid; exact solution `sin(pi x)sin(pi y)`. |
| Boundary penalty sweep | Same 1D equation and initial network as the forward PINN. Compare soft penalties `lambda = 0.01, 0.1, 1, 10, 100, 1000` with a hard-boundary baseline. Every model receives 2,000 Adam updates. | Same 1,001-point grid, plus maximum absolute endpoint error. |
| Deep Ritz | Same 1D equation, initial network, hard boundary factor, and quadrature as the forward PINN. Minimize the energy integral `E(u)=integral(0.5*u'² - f*u)`. | Solution and gradient errors on the same 1,001-point grid. |
| Hybrid inverse problem | Solve `-k*u'' = pi² sin(pi x)` and infer positive `k`. Seed 23; `1–32–32–1` tanh network with hard boundaries; 128 equally spaced interior collocation points. Compare physics plus sensor data with physics alone, starting from identical parameters and `k=1.3`. | True `k=0.7`; exact solution `sin(pi x)/0.7`; relative solution error on 1,000 uniform cell midpoints, separate from training points and sensors. |

All Adam learning rates are `0.001`. The main 1D PINN and Ritz runs receive
2,000 Adam updates followed by 300 L-BFGS iterations. The 2D run receives
2,000 Adam updates followed by up to 600 L-BFGS iterations. Both inverse runs
receive 4,000 Adam updates followed by 40 L-BFGS calls with at most five inner
iterations each, for at most 200 inner iterations. L-BFGS uses a strong Wolfe
line search, so one optimizer iteration may require several loss evaluations.
The history arrays preserve optimizer phases and evaluation counts where
recorded.

The inverse experiment uses 12 sensors equally spaced from `x=0.05` to
`x=0.95`. Synthetic observations add independent Gaussian noise with
`sigma=0.01/0.7=0.0142857`, or 1% of the exact peak amplitude. The hybrid loss
adds five times the mean squared sensor mismatch to the mean squared
normalized PDE residual; the physics-only model sets that data weight to zero.
The parameterization `k=softplus(raw_k)+1e-6` enforces positivity. Exact solution
values enter this experiment only when generating these noisy observations
and evaluating errors. Forward PINN and Ritz training use the forcing and
boundary conditions without exact solution labels.

These are the saved results **after L-BFGS refinement**, except for the
boundary sweep, which uses Adam only.

| Experiment | Selected result |
| --- | --- |
| 1D forward PINN | Relative `L²` error `2.11143e-6`; relative `H¹` seminorm error `1.13998e-5`; normalized residual RMS `5.43211e-5`; endpoint error exactly zero. |
| 2D forward PINN | Relative `L²` error `6.42661e-6`; maximum absolute error `9.44727e-6`; normalized residual RMS over the full evaluation grid `1.24521e-4`; boundary error exactly zero. |
| Boundary sweep, Adam only | Hard-baseline relative `L²` error `1.18545e-4`. The soft model at `lambda=1000` has relative `L²` error `0.612327` under the same budget. See the full sweep below. |
| Deep Ritz | Relative `L²` error `3.85293e-6`; relative `H¹` seminorm error `3.33121e-5`; endpoint error exactly zero. |
| Hybrid inverse problem | Estimated `k=0.700740595`, a `0.105799%` relative coefficient error; held-out relative `L²` error `0.00314372`. Physics-only training gives `k=1.38972734` and relative solution error `0.496304`, despite normalized residual RMS `2.05984e-4`. |

With the same number of Adam updates, changing the boundary weight gives:

| Boundary weight | Relative `L²` error | Maximum absolute endpoint error |
| ---: | ---: | ---: |
| 0.01 | `3.58664e-5` | `4.34621e-5` |
| 0.1 | `1.82574e-4` | `1.88196e-4` |
| 1 | `9.64931e-5` | `3.88686e-5` |
| 10 | `1.14566e-4` | `1.07601e-6` |
| 100 | `1.04621e-3` | `1.50438e-4` |
| 1000 | `6.12327e-1` | `4.05883e-3` |
| Hard boundary | `1.18545e-4` | `0` |

The Ritz/PINN error trajectories and inverse coefficient trajectories in the
figures show **Adam updates only**. Final field plots and final metric values
include subsequent L-BFGS refinement. The boundary sweep, including its hard
baseline, uses Adam only throughout. For example, before refinement the
forward PINN and Ritz relative `L²` errors are `1.18545e-4` and `2.21483e-4`;
their final errors therefore differ from the endpoints of the Adam plots.

The metrics have these definitions:

- Relative `L²` error is `||u_pred-u_exact||_L2 / ||u_exact||_L2`.
  Forward 1D norms use trapezoidal integration on the dense grid. The 2D and
  inverse metrics use Euclidean norms of uniformly sampled values; the
  constant cell weight cancels in the ratio.
- Relative `H¹` **seminorm** error is
  `||u_pred'-u_exact'||_L2 / ||u_exact'||_L2`, evaluated by trapezoidal
  integration for the forward 1D examples. It measures derivative error and
  does not include the solution-value term of the full `H¹` norm.
- The normalized 1D forward residual is `r=(-u''-f)/pi²`; the 2D residual is
  `r=(-Delta u-f)/(2*pi²)`; the inverse residual is `r=(-k*u''-f)/pi²`.
  RMS means the square root of the mean squared residual, with trapezoidal
  integration for forward 1D and sample means for the other problems.
  Normalized residuals and physical residuals have different scales.
- The soft forward loss is the quadrature integral of `r²` plus
  `lambda * (u(0)²+u(1)²)/2`. Loss weights are meaningful only together with
  this normalization.
- The saved forward energy gap uses
  `E(u_pred)-E(u_exact)=0.5*||u_pred'-u_exact'||_L2²`, valid here for trial
  functions satisfying the same Dirichlet conditions. It is evaluated through
  the nonnegative error identity to avoid cancellation.

Generated files are organized as follows:

| Location | Contents |
| --- | --- |
| `results/forward_1d.npz` | Dense coordinates, exact/predicted solutions and derivatives, physical residuals, quadrature nodes and weights, optimizer-phase histories, Adam-only baselines, and sweep arrays. |
| `results/poisson_2d.npz` | Coordinates, exact/predicted fields, absolute error, normalized residual, collocation points, and loss/phase histories. Field arrays are ordered `(y, x)`. |
| `results/inverse_1d.npz` | Sensor data and noise scale, training/evaluation coordinates, both fitted solutions and normalized residuals, coefficient/loss histories, and the analytical solution-family illustration. |
| `results/*.json` | Raw numerical metrics, configurations, seeds, optimizer budgets, software versions, and recorded timing/provenance information. |
| `results/classical_accuracy.npz` and `.json` | Finite-difference and Chebyshev refinement results, nodal residuals, evaluation errors, and the saved PINN reference. |
| `figures/*.pdf` and `figures/*.png` | Ten PDF figures and ten PNG previews, including wave data, the continuous-residual counterexample, and the retained earlier collocation illustration. |
| `snippets/` and `metrics.tex` | Teaching excerpts extracted or assembled from the scripts, reproduction commands, and numerical LaTeX macros generated from saved results. |

The snippets illustrate selected operations and rely on definitions elsewhere
in the complete scripts; they are teaching excerpts rather than standalone
programs. Run the three complete training scripts to reproduce the results.
`make_figures.py` reads the saved arrays and JSON files without retraining.

These runs use simple problems with known solutions and one seed per
configuration. The best boundary weight and the relative performance of PINNs
and Deep Ritz can change with the loss scaling, architecture, sampling,
initialization, and training budget. The inverse physics-only problem is
nonidentifiable: every positive `k` admits the solution `sin(pi x)/k` with the
same forcing and boundary conditions. Its reported coefficient is one
optimizer outcome; the noisy sensors supply the information that distinguishes
coefficients in the hybrid experiment.

`classical_accuracy.py` compares the neural result with two classical solvers. It solves the same 1D Poisson problem in
float64, using centered second-order finite differences with a tridiagonal
solver and Chebyshev polynomial collocation with a dense linear solve. Both
use only forcing and boundary values, not exact solution labels. Finite
differences use linear interpolation; spectral solutions use barycentric
polynomial interpolation. All errors use the original 1,001-point diagnostic
grid, with trapezoidal integration. Some classical nodes coincide with this
grid; it is chosen independently of the solves rather than strictly disjoint.

Doubling the finite-difference resolution reduces the solution error by about
four. With 17 nodes, relative algebraic residual is `6.30e-15`, but relative
solution error is `1.44e-3`: solving a discrete system accurately does not
eliminate discretization error. Spectral degree 16 gives relative solution
error `3.53e-15` on this smooth example. Higher degrees reach a rounding and
conditioning plateau; they do not improve the error monotonically.

This comparison measures accuracy under refinement. Runtime and model size
are not matched. The PINN reference is the saved `2.11e-6` result from one
configuration; other neural methods can reach smaller errors. Specialized neural residual-correction methods can approach
machine precision on selected problems. See [the reading list](limitations_sources.md)
for the sources and scope of the theory and limitations slides.

The analytic illustrations are regenerated by `make_figures.py`, without neural
training. The two illustrations used by the current deck can be run separately:

```sh
python3 experiments/wave_data.py
python3 experiments/continuous_residual.py
```

The first samples the fixed-end wave solution `sin(pi*x)*cos(pi*t)` at three
times. The second evaluates one-neuron constant tanh networks for the
semilinear Neumann problem `-u'' + u/(1+u²) = 0`. The exact continuous residual
loss tends to zero while the `L²` state error diverges. Both norms are obtained
analytically, without a collocation approximation or numerical quadrature.
Their arrays and metadata are saved in `results/wave_data.*` and
`results/continuous_residual.*`.

`counterexample_data.py` provides two optional examples: the tanh
nonattainment sequence and a finite-collocation counterexample. The latter
is included in the saved results but is not used in the slides. See
[the mathematical notes](wellposedness_notes.md) for proofs and the relationship
to the cited coercivity-gap paper.

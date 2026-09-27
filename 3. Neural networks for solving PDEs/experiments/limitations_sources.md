# Reading list: classical solvers and PINN limitations

These references support the comparison with classical solvers, the PINN error
estimates, and the discussion of training difficulties. They also cover
high-accuracy neural methods and applications where neural solvers may help.

1. **Grossmann, T. G., Komorowska, U. J., Latz, J., and Schönlieb, C.-B. (2024).**
   *Can physics-informed neural networks beat the finite element method?*
   IMA Journal of Applied Mathematics 89(1), 143–174.
   [Author manuscript](https://arxiv.org/abs/2302.04107),
   [published paper](https://doi.org/10.1093/imamat/hxae011).
   Compares accuracy and runtime for the PDEs and implementations tested in the paper.

2. **Mishra, S., and Molinaro, R. (2023).**
   *Estimates on the generalization error of physics-informed neural networks for approximating PDEs.*
   IMA Journal of Numerical Analysis 43(1), 1–43.
   [Author manuscript](https://arxiv.org/abs/2006.16144),
   [published paper](https://doi.org/10.1093/imanum/drab093).
   Stability and quadrature connect training error to solution error under assumptions.

3. **De Ryck, T., and Mishra, S. (2024).**
   *Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning.*
   Acta Numerica 33, 633–713.
   [Published article](https://doi.org/10.1017/S0962492923000089).
   Reviews approximation, generalization, and training errors.

4. **Rathore, P., Lei, W., Frangella, Z., Lu, L., and Udell, M. (2024).**
   *Challenges in Training PINNs: A Loss Landscape Perspective.*
   ICML, Proceedings of Machine Learning Research 235, 42159–42191.
   [Official proceedings and PDF](https://proceedings.mlr.press/v235/rathore24a.html).
   Examines differential-operator conditioning and improvements to optimization.

5. **Wang, S., Teng, Y., and Perdikaris, P. (2021).**
   *Understanding and mitigating gradient pathologies in physics-informed neural networks.*
   SIAM Journal on Scientific Computing 43(5), A3055–A3081.
   [Author manuscript](https://arxiv.org/abs/2001.04536),
   [published paper](https://doi.org/10.1137/20M1318043).
   Explains imbalance in gradients from different loss components.

6. **Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., and Mahoney, M. W. (2021).**
   *Characterizing possible failure modes in physics-informed neural networks.*
   NeurIPS 34.
   [Author manuscript](https://arxiv.org/abs/2109.01050).
   Demonstrates training difficulties on convection/reaction/diffusion examples.

7. **Wang, S., Yu, X., and Perdikaris, P. (2022).**
   *When and why PINNs fail to train: A neural tangent kernel perspective.*
   Journal of Computational Physics 449, 110768.
   [Author manuscript](https://arxiv.org/abs/2007.14527).
   Analyzes training rates using a limiting-kernel model under its assumptions.

8. **Wang, Y., and Lai, C.-Y. (2024).**
   *Multi-stage neural networks: Function approximator of machine precision.*
   Journal of Computational Physics 504, 112865.
   [Author manuscript](https://arxiv.org/abs/2307.08934),
   [published paper](https://doi.org/10.1016/j.jcp.2024.112865).
   Shows examples where a neural approximation reaches machine precision.

9. **NumPy reference: floating-point properties.**
   [`numpy.finfo`](https://numpy.org/doc/2.0/reference/generated/numpy.finfo.html).
   `eps` is the representable spacing immediately above 1. It is not a promised
   PDE error, nor the minimum positive representable number.

The classical FEM estimate is Céa's inequality for an exactly solved conforming
Galerkin approximation with a bounded, coercive bilinear form. Reliability of
a posteriori estimators requires further method/problem assumptions; the slides
do not claim such a guarantee for every classical discretization.

The PINN inequality on slide C03 is a conditional illustration, not a theorem
for every PDE: it assumes a suitable stability estimate and a sampling bound
valid for the trained trial function, including the appropriate boundary norms.

The optional classical-accuracy figure is generated from the local Python calculations in
`classical_accuracy.py`, not extracted from a paper. It uses the existing
Poisson problem, float64 arithmetic, second-order finite differences, and
Chebyshev polynomial collocation. The PINN reference is the saved seed-7 run.
There is no equal-cost or equal-parameter-count comparison. Differences near machine precision depend on rounding, conditioning, and the
floating-point reference values.

## Potential advantages of neural PDE methods (slide C11)

- **Han, J., Jentzen, A., and E, W. (2018).** *Solving high-dimensional partial
  differential equations using deep learning.* PNAS 115(34), 8505–8510.
  [Author manuscript](https://arxiv.org/abs/1707.02568),
  [published article](https://doi.org/10.1073/pnas.1718942115).
  Neural BSDE methods demonstrate feasible high-dimensional parabolic examples.
  This supports a potential alternative to tensor-grid growth for structured
  PDEs, not a claim that all PINNs overcome dimensionality or all classical
  methods require tensor grids.
- **Raissi, M., Yazdani, A., and Karniadakis, G. E. (2020).** *Hidden fluid
  mechanics: Learning velocity and pressure fields from flow visualizations.*
  Science 367(6481), 1026–1030.
  [Published article](https://doi.org/10.1126/science.aaw4741),
  [author manuscript](https://arxiv.org/abs/1808.04327).
  Demonstrates hidden-field inference by combining PDE constraints with
  observations. In a joint neural residual fit, parameter updates evaluate the
  trial state without a separate full forward PDE solve. A cost advantage over
  classical inverse methods is problem-dependent; neither poor sensor coverage
  nor the PDE loss guarantees identifiability. Classical all-at-once inverse
  formulations can also avoid nested forward solves.
- **Sukumar, N., and Srivastava, A. (2022).** *Exact imposition of boundary
  conditions with distance functions in physics-informed deep neural networks.*
  Computer Methods in Applied Mechanics and Engineering 389, 114333.
  [Author manuscript](https://arxiv.org/abs/2104.08426),
  [published article](https://doi.org/10.1016/j.cma.2021.114333).
  Geometry-aware trial functions allow meshfree treatment of complex boundaries.
  Reduced meshing/remeshing effort is a potential setup advantage, not a
  demonstrated universal accuracy or wall-clock advantage. Geometry preparation,
  boundary treatment, sampling, and training still contribute to total cost.

For each application on slide C11, compare total cost at matched accuracy.
The papers show what neural methods can do in specific settings; performance
relative to classical solvers depends on the problem and implementation.

# 1.1 Theoretical foundations of FEM

**[Slides: Finite Element Methods, Foundations (PDF, 69 slides)](1.1_fem_foundations.pdf)**
&nbsp;·&nbsp; source [`1.1_fem_foundations.tex`](1.1_fem_foundations.tex)

We follow the Poisson equation, −Δu = f with u = 0 on the boundary, from its weak
form to a finite element solution and an estimate of the error.

| Section | What it covers |
|---|---|
| Motivation and roadmap | why FEM, from a PDE to a simulation, the model problem |
| Weak formulations and well-posedness | integration by parts, the weak Poisson problem, continuity and coercivity, Lax–Milgram, mixed boundary data |
| Galerkin approximation | discrete stability, Galerkin orthogonality, Céa's lemma, best approximation in energy |
| Meshes and finite element spaces | 1D and triangular meshes, mesh quality, Lagrange P<sub>k</sub> elements, hat functions, conforming spaces, reference elements |
| Assembly and discrete systems | element stiffness and load, local-to-global assembly, sparsity, boundary conditions, 2D and 3D fields |
| Error estimates and convergence | interpolation estimates, energy-error rates, reading convergence plots, regularity and the L-shaped domain, L² rates |
| A posteriori estimation and adaptivity | element residuals, flux jumps, error estimators, the adaptive loop, marking |
| Evolution PDEs: space and time | the weak heat equation, semi-discretisation, forward and backward Euler, Crank–Nicolson, stability, balancing space and time errors |
| Checking the solution | checks before trusting a simulation |

## Build

```sh
latexmk -pdf 1.1_fem_foundations.tex
```

The slides use PNGs from [`figures/`](figures), with TikZ diagrams as fallbacks
for missing images. See [`figures/README.md`](figures/README.md) for their sources.

**Next:** [1.2 Computational practice with FEniCS](../1.2.%20Computational%20practice%20with%20FEniCS)

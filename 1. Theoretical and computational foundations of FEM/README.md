# 1. Theoretical and computational foundations of FEM

Start with the mathematics of finite elements, then implement the method in FEniCS.

| Part | Slides | What is inside |
|---|---|---|
| **[1.1 Theoretical foundations](1.1.%20Theoretical%20foundations)** | [69 slides](1.1.%20Theoretical%20foundations/1.1_fem_foundations.pdf) | weak formulations and Lax–Milgram, Galerkin and Céa, Lagrange elements, assembly, a priori and a posteriori error analysis, time stepping |
| **[1.2 Computational practice with FEniCS](1.2.%20Computational%20practice%20with%20FEniCS)** | [Part I, 38 slides](1.2.%20Computational%20practice%20with%20FEniCS/1.2_fenics_part1.pdf) · [Part II, 35 slides](1.2.%20Computational%20practice%20with%20FEniCS/1.2_fenics_part2.pdf) | eleven worked examples in FEniCSx, from Poisson and Stokes to nonlocal PDEs and optimal control, each with a runnable script |

The examples in 1.2 build on the theory in 1.1. You will solve the Poisson problem,
check convergence rates, refine a mesh near a corner, and step the heat equation
forward in time.

**Next chapter:** [2. Fundamentals of Machine Learning](../2.%20Fundamentals%20of%20Machine%20Learning)

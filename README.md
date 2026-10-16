<p align="center">
  <img src=".github/readme/banner.png" alt="From Finite Elements to Neural PDE Solvers: minicourse UFF, CAPES/DAAD" width="100%">
</p>

<p align="center">
  <a href="https://github.com/danielfdzm/minicourse-uff-capes-daad/commits/main"><img src="https://img.shields.io/github/last-commit/danielfdzm/minicourse-uff-capes-daad" alt="GitHub last commit"></a>
  <a href="https://github.com/danielfdzm/minicourse-uff-capes-daad"><img src="https://img.shields.io/github/repo-size/danielfdzm/minicourse-uff-capes-daad" alt="GitHub repo size"></a>
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white" alt="Python 3">
  <img src="https://img.shields.io/badge/FEniCS-legacy%20%26%20FEniCSx-E95420" alt="FEniCS and FEniCSx">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/notebooks-Jupyter-F37626?logo=jupyter&logoColor=white" alt="Jupyter notebooks">
  <img src="https://img.shields.io/badge/slides-305%20pages-B31B1B" alt="305 slides">
</p>

<p align="center">
  <b>Lecture slides, worked experiments and practical exercise notebooks</b><br>
  <a href="#1-theoretical-and-computational-foundations-of-fem">Finite elements</a> &nbsp;·&nbsp;
  <a href="#2-fundamentals-of-machine-learning">Machine learning</a> &nbsp;·&nbsp;
  <a href="#3-neural-networks-for-solving-pdes">Neural networks for PDEs</a> &nbsp;·&nbsp;
  <a href="#getting-started">Getting started</a>
</p>

This minicourse covers finite elements, machine learning, and neural methods for PDEs.
We start with weak formulations and implement them in FEniCS, then learn how to train
neural networks and use them as PDE solvers. The examples compare PINNs and Deep Ritz
with classical methods and show how observations help recover unknown coefficients.

<p align="center">
  <img src=".github/readme/roadmap.png" alt="Course roadmap: 1 Foundations of FEM (1.1 theory, 1.2 FEniCS practice), 2 Fundamentals of machine learning, 3 Neural networks for solving PDEs" width="100%">
</p>

| | Chapter | Slides (PDF) | Hands-on |
|:-:|---|---|---|
| **1.1** | [Theoretical foundations of FEM](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.1.%20Theoretical%20foundations) | [Finite element methods: foundations](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.1.%20Theoretical%20foundations/1.1_fem_foundations.pdf) · 69 slides | [2 labs](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.1.%20Theoretical%20foundations/NOTEBOOK_README.md), [exercises](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.1.%20Theoretical%20foundations/exercises) |
| **1.2** | [Computational practice with FEniCS](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.2.%20Computational%20practice%20with%20FEniCS) | [Part I](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.2.%20Computational%20practice%20with%20FEniCS/1.2_fenics_part1.pdf) · 38 slides<br>[Part II](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.2.%20Computational%20practice%20with%20FEniCS/1.2_fenics_part2.pdf) · 35 slides | 14 Python examples, [lab](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.2.%20Computational%20practice%20with%20FEniCS/NOTEBOOK_README.md), [exercises](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.2.%20Computational%20practice%20with%20FEniCS/exercises) |
| **2** | [Fundamentals of Machine Learning](2.%20Fundamentals%20of%20Machine%20Learning) | [Neural networks for supervised regression](2.%20Fundamentals%20of%20Machine%20Learning/2_machine_learning_fundamentals.pdf) · 80 slides | [3 labs](2.%20Fundamentals%20of%20Machine%20Learning/NOTEBOOK_README.md), [exercises](2.%20Fundamentals%20of%20Machine%20Learning/exercises) |
| **3** | [Neural networks for solving PDEs](3.%20Neural%20networks%20for%20solving%20PDEs) | [From data-driven regression to PINNs, Deep Ritz and hybrid losses](3.%20Neural%20networks%20for%20solving%20PDEs/3_neural_networks_for_pdes.pdf) · 83 slides | slide experiments, [7 labs](3.%20Neural%20networks%20for%20solving%20PDEs/NOTEBOOK_README.md), [exercises](3.%20Neural%20networks%20for%20solving%20PDEs/exercises) |

---

## 1. Theoretical and computational foundations of FEM

<p align="center">
  <a href=".github/readme/fem_eigenmode_still.png">
    <img src=".github/readme/fem_eigenmode.gif" alt="Animation: P1 finite element approximation of the first eigenfunction of the L-shaped domain on successively refined meshes, with the eigenvalue error falling" width="100%">
  </a>
</p>
<p align="center"><sub>P1 finite elements for the first Dirichlet eigenfunction of the L-shaped domain. The eigenvalue error falls with every refinement, at a rate limited by the re-entrant corner.</sub></p>

**[1.1 Theoretical foundations](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.1.%20Theoretical%20foundations)**
&nbsp;·&nbsp; We use the Poisson equation to introduce weak formulations, Galerkin
approximation, finite element spaces, and assembly. Error estimates lead to adaptive
refinement; the heat equation then introduces time stepping.

**[1.2 Computational practice with FEniCS](1.%20Theoretical%20and%20computational%20foundations%20of%20FEM/1.2.%20Computational%20practice%20with%20FEniCS)**
&nbsp;·&nbsp; Implement the weak forms and check the computed solutions. *Part I* covers
Poisson convergence tests, Neumann problems with local refinement, the
L-shaped eigenproblem, the heat equation, a nonlinear Poisson equation and Stokes flow.
*Part II* covers a 3D p-Laplacian, nonlocal parabolic PDEs, degenerate null control,
PDE-constrained optimal control and a coupled reaction–diffusion system. Every example
has a script in `experiments/`.

The laboratory notebooks make assembly, convergence, adaptive refinement and
heat stepping visible in small numerical experiments. The FEniCSx laboratory then
implements Poisson and heat problems using the same environment as the scripts.
Each part has an `exercises/` folder with section-by-section practical tasks.

<p align="center">
  <img src=".github/readme/chapter1_highlights.png" alt="Chapter 1 highlights: tetrahedral meshes of the unit cube, Stokes flow in a lid-driven cavity, Gray–Scott reaction–diffusion patterns" width="100%">
</p>

## 2. Fundamentals of Machine Learning

<p align="center">
  <a href=".github/readme/ml_training_still.png">
    <img src=".github/readme/ml_training.gif" alt="Animation: a small neural network fits noisy samples; training error keeps falling while validation error turns back up" width="100%">
  </a>
</p>
<p align="center"><sub>A 1–48–48–1 tanh network trained with Adam on 40 noisy samples. Training error keeps falling after validation error starts to rise. Early stopping keeps the checkpoint with the lowest validation error.</sub></p>

**[Neural networks for supervised regression](2.%20Fundamentals%20of%20Machine%20Learning/2_machine_learning_fundamentals.pdf)**
&nbsp;·&nbsp; Population and empirical risk, linear models, neural networks as function
classes and the universal approximation theorems (Cybenko, Pinkus), gradient descent and
backpropagation, regularisation and early stopping, width and depth, and validation and
testing without leakage.

**[Machine Learning Laboratory](2.%20Fundamentals%20of%20Machine%20Learning/NOTEBOOK_README.md)**
&nbsp;·&nbsp; Three worked notebooks follow the slide dataset from least squares to
neural approximation, optimisation and model selection. The companion
[practicals](2.%20Fundamentals%20of%20Machine%20Learning/exercises) ask you to change
the experiments, check gradients and compare validation results before testing.

<p align="center">
  <img src=".github/readme/chapter2_highlights.png" alt="Chapter 2 highlights: from biological to artificial neurons, universal approximation with wider layers, underfitting and overfitting" width="100%">
</p>

## 3. Neural networks for solving PDEs

<p align="center">
  <a href=".github/readme/pinn_training_still.png">
    <img src=".github/readme/pinn_training.gif" alt="Animation: a physics-informed neural network learns the solution of a 2D Poisson problem; its pointwise error and training curves shrink" width="100%">
  </a>
</p>
<p align="center"><sub>A PINN for −Δu = 2π² sin(πx) sin(πy) on the unit square, trained only on the PDE residual at 1,024 points, with the boundary condition built into the network. After 3,000 Adam steps the relative L² error is about 7·10⁻⁵.</sub></p>

**[From data-driven regression to PINNs, Deep Ritz and hybrid losses](3.%20Neural%20networks%20for%20solving%20PDEs/3_neural_networks_for_pdes.pdf)**
&nbsp;·&nbsp; Strong-form PINNs (collocation, quadrature, stability and error, training
limits), network design and exact boundary conditions, Deep Ritz and weak formulations,
hybrid PDE–data losses and inverse problems. We also examine the training difficulties
and compare with classical solvers. The scripts in `experiments/` reproduce the slide results.

**[Neural PDE Laboratory](3.%20Neural%20networks%20for%20solving%20PDEs/NOTEBOOK_README.md)**
&nbsp;·&nbsp; Run 13 experiments, inspect the errors, and try changing the setup. Topics include
a vibrating membrane, sensor placement, Gray–Scott patterns, Fourier features, adaptive
sampling, heat flow, and a network trained for several PDE parameters. Seven independent
notebooks include saved static plots for convenient GitHub previews. To browse the
figures and seven interactive plots, download
`notebook_outputs/` and open `index.html` in a browser.

The [practical exercises](3.%20Neural%20networks%20for%20solving%20PDEs/exercises)
cover all six lecture sections, from putting a PDE into a loss to checking a trained
solution on independent points. Each includes a working baseline and tasks to extend it.

<p align="center">
  <img src=".github/readme/chapter3_highlights.png" alt="Chapter 3 highlights: a 2D Poisson PINN with its error and residual, a mesh-free solution on an annulus, an inverse problem's solution family and valley" width="100%">
</p>

---

## Repository layout

```text
.
├── 1. Theoretical and computational foundations of FEM/
│   ├── 1.1. Theoretical foundations/
│   │   ├── 1.1_fem_foundations.tex / .pdf        slides
│   │   ├── FEM_01_*.ipynb / FEM_02_*.ipynb      worked numerical experiments
│   │   ├── NOTEBOOK_README.md                    notebook guide and setup
│   │   ├── requirements-notebook.txt             NumPy/SciPy notebook environment
│   │   ├── exercises/                            practical tasks and README
│   │   └── figures/                              figures of the slides
│   └── 1.2. Computational practice with FEniCS/
│       ├── 1.2_fenics_part1.tex / .pdf           Part I: from weak forms to working code
│       ├── 1.2_fenics_part2.tex / .pdf           Part II: advanced PDE examples
│       ├── environment.yml                       FEniCSx environment
│       ├── FEniCSx_01_poisson_and_heat.ipynb     worked FEniCSx experiment
│       ├── NOTEBOOK_README.md                    notebook guide and setup
│       ├── requirements-notebook.txt             notebook tools for that environment
│       ├── exercises/                            Part I and Part II practicals
│       └── experiments/                          example scripts 01–14
│           ├── figures/                          their plots, used by the slides
│           └── results/                          XDMF/HDF5 fields, CSV tables, run logs
├── 2. Fundamentals of Machine Learning/
│   ├── 2_machine_learning_fundamentals.tex / .pdf
│   ├── ML_01_*.ipynb … ML_03_*.ipynb            three worked experiments
│   ├── NOTEBOOK_README.md                        notebook guide and setup
│   ├── requirements-notebook.txt                 machine-learning notebook environment
│   ├── exercises/                                three practical notebooks and README
│   └── figures/
└── 3. Neural networks for solving PDEs/
    ├── 3_neural_networks_for_pdes.tex / .pdf
    ├── experiments/                              scripts, results and figures behind the slides
    ├── Neural_PDE_Laboratory.ipynb               laboratory index
    ├── Neural_PDE_01_*.ipynb … Neural_PDE_07_*.ipynb  seven executed lab notebooks
    ├── exercises/                                six section practicals and README
    └── notebook_outputs/                         figures and interactive HTML explorers
```

Each chapter folder has its own README with more detail.

Worked notebooks sit beside the slides. Practical notebooks and their section
coverage tables live in each chapter or part's `exercises/` folder. Slide scripts,
figures and saved laboratory outputs retain their existing locations.

## Getting started

**Read the slides.** The PDFs sit next to their LaTeX sources (see the table above).

**Run the FEM foundations laboratory** (Chapter 1.1, Python 3.11+):

```sh
cd "1. Theoretical and computational foundations of FEM/1.1. Theoretical foundations"
python -m pip install -r requirements-notebook.txt
python -m jupyter lab FEM_01_assembly_and_convergence.ipynb
```

**Run the machine-learning laboratory** (Chapter 2):

```sh
cd "2. Fundamentals of Machine Learning"
python -m pip install -r requirements-notebook.txt
python -m jupyter lab ML_01_regression_and_risk.ipynb
```

Run each command block from the repository root. In Jupyter, select **Restart Kernel
and Run All Cells** to reproduce a baseline. Open the notebooks in `exercises/` for
the practical tasks: record your prediction, edit the supplied baseline, then explain
the measured result. The exercise READMEs map every lecture section to its tasks.

**Rebuild a deck.** From the deck's folder, run `latexmk -pdf <deck>.tex`. This needs a TeX
distribution with Beamer, TikZ/PGFPlots, `listings` and `lmodern`; the FEniCS decks also
use `appendixnumberbeamer`.

**Run the FEniCSx examples** (Chapter 1.2):

```sh
cd "1. Theoretical and computational foundations of FEM/1.2. Computational practice with FEniCS"
conda env create -f environment.yml
conda activate fenicsx-env
python experiments/01_poisson.py
# Add Jupyter to the same environment to run the laboratory and practicals:
python -m pip install -r requirements-notebook.txt
python -m jupyter lab FEniCSx_01_poisson_and_heat.ipynb
```

The scripts write to `experiments/figures/` and `experiments/results/` from any working
directory. On a machine without a display, set `PYVISTA_OFF_SCREEN=true` first.

**Run the neural-PDE laboratory** (Chapter 3):

```sh
cd "3. Neural networks for solving PDEs"
python -m pip install -r experiments/requirements-notebook.txt
python -m jupyter lab Neural_PDE_01_foundations.ipynb
```

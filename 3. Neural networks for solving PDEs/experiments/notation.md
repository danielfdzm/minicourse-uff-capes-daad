# Shared notation for the slides

These conventions apply to the equations, captions, and generated figure labels.
Code retains descriptive identifiers such as `model`, `u`, `k`, and `raw_k`;
inside a training function, `u` stores network values, not the exact solution.

## States, networks, and dimensions

| Symbol | Meaning |
| --- | --- |
| $x\in\Omega\subset\mathbb R^d$, $t$ | Spatial coordinate and dimension; time when present. |
| $u:\Omega\to\mathbb R^{d_u}$ | Exact PDE solution; $d_u$ is the output dimension. |
| $u_\theta$, $\theta\in\Theta\subset\mathbb R^P$ | Neural approximation; $P$ counts trainable parameters. |
| $u_h$, $\widetilde u_h$ | Classical discrete solution; its approximation after an inexact algebraic solve. |
| $\widehat\theta$, $u_{\widehat\theta}$ | Parameters returned by training and their network. The hat denotes the training result; it need not be a global minimizer. |
| $\kappa$, $\widehat\kappa$ | True coefficient and its fitted estimate in the inverse experiment. A variable candidate coefficient is identified explicitly where a solution family is explored. |
| $m$, $s$ | Number of hidden neurons in the local tanh example; differential-operator order. Neither is the output dimension or parameter count. |
| $N_\theta$, $\rho$, $G$ | Raw network, boundary-vanishing factor, and extension of Dirichlet data; the constrained ansatz is $u_\theta=G+\rho N_\theta$. |
| $v$, $v_n$ | Generic candidate or test function as specified locally; $v_n$ denotes the explicit neural sequence in each counterexample. |

For the scalar shallow tanh example of width $m$, $P=3m+1$. Elsewhere network
architecture determines $P$. The discrepancy function $m_\eta$ is labeled as
such where introduced; its symbol does not count neurons.

## Objectives and samples

| Symbol | Meaning |
| --- | --- |
| $\mathcal J$, $\widehat{\mathcal J}$ | Continuous objective and its sampled or quadrature approximation. In LaTeX these are `\Loss` and `\widehat\Loss`. |
| $\mathcal E$, $\widehat{\mathcal E}$ | Variational energy and quadrature energy (`\Energy`, `\widehat\Energy`). |
| $\mathbb E$ | Expectation (`\E`), distinct from energy. |
| $r_\theta=\mathcal L u_\theta-f$, $b_\theta=\mathcal B u_\theta-g$ | Strong PDE and boundary residuals. Bold $\mathbf r(\theta)$ collects sampled residuals into a vector. |
| $x_i^r$, $x_i^b$, $x_i^d$ | Interior-residual, boundary, and observation locations. |
| $N_r$, $N_b$, $N_d$ | Corresponding sample counts; $y_i$ is the observation at $x_i^d$. |
| $\lambda_r$, $\lambda_b$, $\lambda_d$ | Weights of residual, boundary, and data terms. |
| $w_i^r$, $w_i^b$ | Quadrature weights; superscripts indicate the point family. |

We write $\mathcal J(\theta)=\mathcal J(u_\theta)$ when treating a functional
as an objective on parameter space. The same convention applies to sampled
losses. Empirical data losses carry a hat.

Quadrature weights may already include domain volume or boundary measure.
A mean loss uses weights $1/N$, while weights $|\Omega|/N$ approximate an
integral under uniform spatial sampling. Overall factors can be absorbed into
the term weights $\lambda_r,\lambda_b,\lambda_d$; reported weights must always
be read with the stated residual scaling and normalization.

A finite sum of **exact weak moments** is not empirical merely because the
test set is finite. Its loss can remain $\mathcal J_{\rm weak}$; replacing the
spatial integrals by quadrature gives $\widehat{\mathcal J}_{\rm weak}$.

## Weak forms, inverse problems, and error norms

| Symbol | Meaning |
| --- | --- |
| $\mathcal R_\theta(v)=a(u_\theta,v)-F(v)$ | Weak residual applied to a test function; `\WeakRes` in LaTeX. |
| $A(x)$ | Diffusion tensor in the PDE. |
| $\mathcal A:V\to V'$ | Variational operator, defined by $\langle\mathcal A w,v\rangle=a(w,v)$. |
| $\mathcal P$ | Regularization functional (`\Reg`); distinct from parameter count $P$. |
| $\mathcal H$ | Observation operator (`\Obs`); distinct from Sobolev spaces such as $H^1$. |
| $\Sigma_d$, $\sigma_d^2$ | Positive-definite observation-noise covariance; scalar observation-noise variance when applicable. |

The covariance-weighted norm is $\|z\|_{\Sigma_d^{-1}}^2=z^T\Sigma_d^{-1}z$.
State-error norms name their space, for example $\|u_\theta-u\|_{L^2(\Omega)}$.
The counterexamples use $L^2(0,1)$. Relative errors divide by the corresponding
exact-solution norm; an $H^1$ **seminorm** measures derivatives only.

For the counterexample proofs and references, see
[wellposedness_notes.md](wellposedness_notes.md).

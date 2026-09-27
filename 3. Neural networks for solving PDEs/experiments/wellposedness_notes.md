# Existence and convergence: two different questions

These notes give the proofs for slides L3-Q01–L3-Q05. The examples separate
two questions: whether a neural minimizer exists, and whether a small
residual implies a small solution error. See the [notation guide](notation.md)
for the symbols used here.

Source: Enrique Zuazua, *The Coercivity Gap in Neural PDE Solvers: Parameter
Escape and State Convergence*, [arXiv:2606.04018v2](https://arxiv.org/html/2606.04018v2),
27 June 2026. The arXiv abstract page retains “Functional Convergence” in its
title; the v2 full text says “State Convergence.”

The paper's worked nonattainment example uses Gaussian neurons, not tanh.
The first tanh example below illustrates nonclosed realization classes. The
second is an independent continuous-residual counterexample for a semilinear
Neumann problem. Neither construction is quoted from the paper.

## A tanh infimum that is not attained

The boundary-value problem $-u''=0$, $u(0)=0$, $u(1)=1$ has the unique solution
$u(x)=x$. Fix a finite positive width $m$ and consider

$$
u_\theta(x)=c+\sum_{j=1}^m a_j\tanh(b_jx+d_j),
\qquad \theta\in\Theta=\mathbb R^P,\quad P=3m+1.
$$

All parameters are finite real numbers and there is no linear skip connection.
For a generic candidate $v$, use the continuous residual loss plus squared
endpoint mismatch:

$$
\mathcal J(v)=\int_0^1|v''(x)|^2\,dx+|v(0)|^2+|v(1)-1|^2.
$$

The one-neuron sequence $v_n(x)=\tanh(x/n)/\tanh(1/n)$ lies in every such class
with $m\ge1$ by setting additional output weights to zero. It satisfies the
boundary values exactly. Taylor expansion, uniformly with two derivatives on
$[0,1]$, gives

$$
v_n(x)=x+\frac{x-x^3}{3n^2}+O(n^{-4}),\qquad
v_n''(x)=-\frac{2x}{n^2}+O(n^{-4}).
$$

Consequently $\mathcal J(v_n)=4/(3n^4)+O(n^{-6})$ tends to zero. Its output
weight $a_1^{(n)}=1/\tanh(1/n)$ grows like $n$.

If a finite network $u_\theta$ achieved loss zero, it would equal $x$ on $(0,1)$.
A finite tanh sum is real analytic on the whole real line, so the identity
theorem would force equality with $x$ everywhere. That is impossible because a
finite tanh sum is bounded on the real line. Thus

$$
\inf_{\theta\in\Theta}\mathcal J(u_\theta)=0,\qquad
\operatorname*{argmin}_{\theta\in\Theta}\mathcal J(u_\theta)=\varnothing.
$$

The states nevertheless converge to the exact solution in $C^2([0,1])$.
Adding a linear skip connection removes this particular example.

The conclusion depends on the architecture. For the broader discussion, see
[Sections 2 and 4](https://arxiv.org/html/2606.04018v2#S4) and its
[architecture discussion in Section 7](https://arxiv.org/html/2606.04018v2#S7).

## Continuous residuals vanish while neural states diverge

The second example uses the exact continuous loss. Consider

$$
-u''+\frac{u}{1+u^2}=0\quad\text{in }(0,1),\qquad u'(0)=u'(1)=0.
$$

Multiplying by $u$ and integrating by parts gives

$$
\int_0^1\left(|u'|^2+\frac{u^2}{1+u^2}\right)\,dx=0.
$$

Both integrands are nonnegative, so the unique exact solution is $u=0$.
Define the continuous strong-residual loss with its Neumann penalties by

$$
\mathcal J(v)=\int_0^1\left|-v''+\frac{v}{1+v^2}\right|^2\,dx
+|v'(0)|^2+|v'(1)|^2.
$$

The constant sequence $v_n(x)=n$ is exactly representable by one tanh neuron,
even without an output bias:

$$
v_n(x)=\frac{n}{\tanh(1)}\tanh(0x+1).
$$

Its derivatives and Neumann boundary residuals vanish. The interior residual
is the constant $n/(1+n^2)$. Hence the **exact integrals** give

$$
\mathcal J(v_n)=\frac{n^2}{(1+n^2)^2}\longrightarrow0,
\qquad
\|v_n-u\|_{L^2(0,1)}=n\longrightarrow\infty.
$$

The neural minimum is attained at the zero network, yet this other minimizing
sequence escapes in state space. Thus attainment and uniqueness do not imply
that every minimizing sequence converges.

The cause is lack of a **global residual-to-state stability estimate**: the
reaction $s/(1+s^2)$ tends to zero as $s$ tends to infinity. The PDE has a
unique zero-data solution, but this example does not claim global well-posedness
under arbitrary forcing perturbations. It does not contradict residual error
bounds for stable problems such as linear Dirichlet Poisson in the appropriate
norms. The example is independent of any sampling rule and is not specific to
neural solvers; a tanh network simply represents the sequence exactly.

`continuous_residual.py` evaluates these analytic formulas and exports the
figure and numeric values. No numerical quadrature is needed to obtain either
norm. The residual curve is $\sqrt{\mathcal J(v_n)}$, while the state curve is
$\|v_n-u\|_{L^2(0,1)}$.

`counterexample_data.py` and its saved figure provide a separate
finite-collocation example, which is not used in the slides.

For the distinction between residual norms that control states and residuals
that do not, see the paper's
[Section 2.4](https://arxiv.org/html/2606.04018v2#S2.SS4) and
[Section 8.1](https://arxiv.org/html/2606.04018v2#S8.SS1).

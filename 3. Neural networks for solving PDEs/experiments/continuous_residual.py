#!/usr/bin/env python3
"""Plot an exact continuous-residual counterexample, with no training.

Run from the project root: python3 experiments/continuous_residual.py

The semilinear Neumann problem is
    -u'' + u/(1 + u**2) = 0,  x in (0, 1),  u'(0) = u'(1) = 0.
Its unique exact solution is zero. A single tanh neuron realizes v_n(x) = n:
    v_n(x) = (n/tanh(1)) * tanh(0*x + 1).
The exact, integrated strong-residual norm is n/(1+n**2), whereas the
L2 solution error is n. No collocation or quadrature defines these metrics.

This is a failure of global residual-to-state coercivity for a saturating
reaction. It does not contradict stability estimates for coercive PDEs.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, TEAL, ORANGE = "#193C55", "#117479", "#CE6F24"
GREY = "#647786"


def constant_network(x, n):
    """One tanh neuron, zero input weight, hidden bias one, no output bias."""
    return (n / np.tanh(1.0)) * np.tanh(0.0 * np.asarray(x) + 1.0)


def main():
    n = 2.0 ** np.arange(7)
    x = np.linspace(0.0, 1.0, 301)  # Plot coordinates only.
    network_u = np.array([constant_network(x, value) for value in n])
    # Both derivatives vanish identically because the neuron's input weight is 0.
    first_derivative = np.zeros_like(network_u)
    second_derivative = np.zeros_like(network_u)
    boundary_derivative = np.zeros((len(n), 2))
    residual = -second_derivative + network_u / (1.0 + network_u**2)
    # The integrands are constant and the interval has length one. These are
    # closed-form exact integrals, not values obtained using the plot grid.
    residual_l2 = n / (1.0 + n**2)
    continuous_loss = residual_l2**2
    solution_l2_error = n.copy()
    np.testing.assert_allclose(network_u, np.broadcast_to(n[:, None], network_u.shape),
                               rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(residual,
                               np.broadcast_to(residual_l2[:, None], residual.shape),
                               rtol=2e-15, atol=0.0)
    np.testing.assert_array_equal(first_derivative, np.zeros_like(first_derivative))
    np.testing.assert_array_equal(boundary_derivative, np.zeros((len(n), 2)))
    assert np.all(np.diff(residual_l2) < 0.0)
    assert np.all(np.diff(solution_l2_error) > 0.0)

    figures, results = ROOT / "figures", ROOT / "results"
    figures.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    np.savez_compressed(
        results / "continuous_residual.npz", n=n, plot_x=x,
        network_u=network_u, boundary_derivative=boundary_derivative,
        exact_integrated_residual_l2=residual_l2,
        exact_continuous_loss=continuous_loss,
        exact_integrated_solution_l2_error=solution_l2_error,
    )
    metadata = {
        "equation": "-u'' + u/(1+u^2) = 0 on (0,1)",
        "boundary_conditions": "u'(0)=u'(1)=0",
        "unique_exact_solution": "u=0",
        "uniqueness_proof": "Multiply by u and integrate: integral(|u'|^2 + u^2/(1+u^2)) = 0.",
        "neural_sequence": "v_n(x) = (n/tanh(1))*tanh(0*x+1) = n",
        "continuous_loss": "integral_0^1|-v_n''+v_n/(1+v_n^2)|^2 dx + |v_n'(0)|^2 + |v_n'(1)|^2",
        "residual_l2_formula": "n/(1+n^2)",
        "continuous_loss_formula": "n^2/(1+n^2)^2",
        "solution_l2_error_formula": "n",
        "metric_computation": "Closed-form continuous integrals; no quadrature or collocation.",
        "plot_grid_role": "Used only to draw the spatial profiles.",
        "training_performed": False,
        "boundary_derivative_max_abs": float(np.max(np.abs(boundary_derivative))),
        "scope": "Unique exact solution, but no global coercivity; no claim of stability for arbitrary perturbed data.",
        "n": n.tolist(),
        "residual_l2": residual_l2.tolist(),
        "continuous_loss_values": continuous_loss.tolist(),
        "solution_l2_error": solution_l2_error.tolist(),
    }
    (results / "continuous_residual.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": NAVY, "axes.labelcolor": NAVY,
        "xtick.color": GREY, "ytick.color": GREY,
        "axes.edgecolor": "#BAC8CF", "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "#F4F6F7", "figure.facecolor": "white",
        "grid.color": "#E3EAED", "grid.linewidth": 0.7,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
        "lines.linewidth": 2.0,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 2.5))
    fig.subplots_adjust(left=0.072, right=0.985, bottom=0.22, top=0.83, wspace=0.40)
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major")
        ax.tick_params(length=3, labelsize=10)

    left.set_title("A   The neural states diverge", loc="left", pad=10,
                   fontsize=11.5, fontweight="bold")
    for value, color in [(1, GREY), (4, TEAL), (16, ORANGE)]:
        left.plot(x, constant_network(x, value), color=color, label=rf"$n={value}$")
    left.axhline(0.0, color=NAVY, ls=(0, (4, 3)), lw=1.5, label="Exact: $u=0$")
    left.set(xlabel="Position $x$", ylabel="$v_n(x)$", xlim=(0, 1),
             ylim=(-1.5, 18), xticks=[0, 0.5, 1], yticks=[0, 4, 8, 12, 16])
    left.legend(loc="upper left", bbox_to_anchor=(0.02, 0.75), ncol=2,
                fontsize=9.4, handlelength=1.9, columnspacing=1.2)

    right.set_title("B   The continuous residual vanishes", loc="left", pad=10,
                    fontsize=11.5, fontweight="bold")
    right.loglog(n, solution_l2_error, "s-", color=ORANGE, ms=5.5,
                 markeredgecolor="white", label=r"Solution error $\|v_n-u\|_{L^2}$")
    right.loglog(n, residual_l2, "o-", color=TEAL, ms=5.5,
                 markeredgecolor="white", label=r"Residual $\|r(v_n)\|_{L^2}$")
    right.set_xscale("log", base=2)
    right.set_xticks([1, 4, 16, 64], labels=["1", "4", "16", "64"])
    right.set(xlabel="Network amplitude $n$", ylabel="Exact $L^2$ norm",
              xlim=(0.85, 76), ylim=(0.01, 110), yticks=[0.01, 0.1, 1, 10, 100])
    right.legend(loc="upper left", fontsize=9.2, handlelength=1.8,
                 borderaxespad=0.25, labelspacing=0.45)
    fig.savefig(figures / "continuous_residual.pdf")
    fig.savefig(figures / "continuous_residual.png", dpi=240)
    plt.close(fig)
    print("Saved experiments/figures/continuous_residual.pdf and continuous_residual.png")
    print("Exact integrated metrics; no collocation, quadrature, or training.")
    print(f"n=64: residual L2={residual_l2[-1]:.9f}; loss={continuous_loss[-1]:.9f}; "
          f"solution L2 error={solution_l2_error[-1]:.1f}; boundary derivative=0.")


if __name__ == "__main__":
    main()

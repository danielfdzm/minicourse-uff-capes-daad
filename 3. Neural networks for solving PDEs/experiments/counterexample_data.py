#!/usr/bin/env python3
"""Compute and plot two explicit tanh-network counterexamples.

Run from the project root: python3 experiments/counterexample_data.py
No training is involved: the network parameters are given analytically.

The examples illustrate nonattainment in a fixed tanh class and the failure of
fixed-point residual sampling to control the continuous PDE solution error.
They are worked examples, not examples quoted verbatim from Zuazua's paper.
"""
from pathlib import Path
import json

import numpy as np

from make_figures import GREY, NAVY, ORANGE, TEAL, pair, save, title

ROOT = Path(__file__).resolve().parent
A = 1.0 / 3.0
B = 1.0 - A


def sech_squared(z):
    """Evaluate sech(z)^2 without the cancellation in 1 - tanh(z)^2."""
    q = np.exp(-2.0 * np.abs(z))
    return 4.0 * q / (1.0 + q) ** 2


def bump(x, k):
    """A two-neuron tanh network with an output bias and zero endpoints."""
    c = np.tanh(k * B) - np.tanh(k * A)
    return (np.tanh(k * (x - A)) - np.tanh(k * (x - B)) - c) / 2.0


def bump_second_derivative(x, k):
    za, zb = k * (x - A), k * (x - B)
    return k**2 * (-sech_squared(za) * np.tanh(za)
                   + sech_squared(zb) * np.tanh(zb))


def affine_approximation(x, n):
    """A one-neuron tanh network with endpoint values zero and one."""
    return np.tanh(x / n) / np.tanh(1.0 / n)


def affine_second_derivative(x, n):
    z = x / n
    return -2.0 * sech_squared(z) * np.tanh(z) / (n**2 * np.tanh(1.0 / n))


def l2_norm(values, x):
    return np.sqrt(np.trapezoid(values**2, x))


def derivative_check(function, derivative, x, h):
    """Compare analytic derivatives with an independent five-point stencil."""
    finite_difference = (-function(x + 2 * h) + 16 * function(x + h)
                         - 30 * function(x) + 16 * function(x - h)
                         - function(x - 2 * h)) / (12 * h**2)
    expected = derivative(x)
    return float(np.max(np.abs(finite_difference - expected))
                 / np.max(np.abs(expected)))


def draw(data):
    fig, (left, right) = pair()
    title(left, "A   The networks stay wrong")
    for k, color in [(16, GREY), (64, TEAL), (256, ORANGE)]:
        index = int(np.flatnonzero(data["k"] == k)[0])
        left.plot(data["x"], data["bump_u"][index], color=color,
                  label=rf"$k={k}$", lw=2.3)
    left.axhline(0.0, color=NAVY, ls=(0, (4, 3)), lw=1.8,
                   label="Exact solution: 0")
    for x_i in data["collocation_x"]:
        left.axvline(x_i, color=GREY, ls=":", lw=1.0, alpha=0.65, zorder=0)
    left.plot(data["collocation_x"], np.full(3, -0.12), "|", color=NAVY,
              ms=10, markeredgewidth=1.6)
    left.text(0.5, -0.245, "Dotted lines: residual sample locations",
              ha="center", va="center", fontsize=9, color=GREY)
    left.set(xlabel="Position $x$", ylabel="$w_k(x)$", xlim=(0, 1),
              ylim=(-0.31, 1.42), yticks=[0, 0.5, 1])
    left.legend(loc="upper left", ncol=2, fontsize=9, columnspacing=0.8,
                 handlelength=1.7, borderaxespad=0.3)

    title(right, "B   The sampled residual vanishes")
    right.semilogy(data["k"], data["sampled_residual_rms"], "o-",
                    color=TEAL, ms=5.5, markeredgecolor="white",
                    label=r"Sampled residual RMS")
    right.semilogy(data["k"], data["bump_l2_error"], "s-", color=ORANGE,
                    ms=5.5, markeredgecolor="white", label=r"Solution error $L^2$")
    right.set_xscale("log", base=2)
    right.set_xticks(data["k"], labels=[str(k) for k in data["k"]])
    right.set(xlabel="Steepness $k$", ylabel="Norm (log scale)",
               xlim=(6.5, 310), ylim=(1e-14, 1e2),
               yticks=[1e-12, 1e-8, 1e-4, 1])
    right.legend(loc="lower left", fontsize=9.5)
    right.annotate(r"Error $\to 1/\sqrt{3}$", (256, data["bump_l2_error"][-1]),
                    xytext=(-6, 12), textcoords="offset points", ha="right",
                    color=ORANGE, fontsize=10)
    save(fig, "residual_counterexample")


def main():
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    x = np.linspace(0.0, 1.0, 65537)
    x_fine = np.linspace(0.0, 1.0, 131073)
    k_values = np.array([8, 16, 32, 64, 128, 256])
    collocation = np.array([0.25, 0.5, 0.75])
    n_values = np.array([1, 2, 4, 8, 16, 32])
    bump_u = np.array([bump(x, k) for k in k_values])
    sampled_residual = np.array([-bump_second_derivative(collocation, k)
                                 for k in k_values])
    rms = np.sqrt(np.mean(sampled_residual**2, axis=1))
    bump_l2 = np.array([l2_norm(v, x) for v in bump_u])
    continuous_rms = np.array([l2_norm(bump_second_derivative(x, k), x)
                               for k in k_values])
    affine_u = np.array([affine_approximation(x, n) for n in n_values])
    affine_residual = np.array([affine_second_derivative(x, n) for n in n_values])
    affine_residual_l2 = np.array([l2_norm(v, x) for v in affine_residual])
    affine_l2_error = np.array([l2_norm(v - x, x) for v in affine_u])
    boundary_bump = float(np.max(np.abs(bump_u[:, [0, -1]])))
    boundary_affine = float(np.max(np.abs(affine_u[:, [0, -1]] - [0.0, 1.0])))
    refinement_difference = max(abs(l2_norm(bump(x_fine, k), x_fine) - norm)
                                for k, norm in zip(k_values, bump_l2))
    derivative_errors = [derivative_check(
        lambda z: bump(z, k), lambda z: bump_second_derivative(z, k),
        np.array([A + 0.5 / k, B - 0.8 / k]), 0.001 / k)
        for k in k_values]
    affine_derivative_errors = [derivative_check(
        lambda z: affine_approximation(z, n),
        lambda z: affine_second_derivative(z, n),
        np.array([0.2, 0.4, 0.8]), 0.002) for n in n_values]
    # A separate high-k check confirms the predicted nonzero L2 limit.
    limit_l2 = np.sqrt(B - A)
    limit_check_k = 4096
    limit_check_l2 = float(l2_norm(bump(x_fine, limit_check_k), x_fine))
    assert boundary_bump < 1e-14 and boundary_affine < 1e-14
    assert refinement_difference < 1e-9
    assert max(derivative_errors) < 2e-8
    assert max(affine_derivative_errors) < 2e-6
    assert np.all(rms > 0.0), "Do not replace analytic residuals with a display floor."
    assert rms[-1] < 1e-12 and bump_l2[-1] > 0.57
    assert abs(limit_check_l2 - limit_l2) < 3e-4
    assert np.all(np.diff(affine_residual_l2) < 0)
    assert np.all(np.diff(affine_l2_error) < 0)
    # Taylor expansion gives n^4 * integral |v_n''|^2 -> 4/3.
    scaled_affine_loss = affine_residual_l2**2 * n_values**4
    assert abs(scaled_affine_loss[-1] - 4.0 / 3.0) < 0.003

    data = {
        "x": x, "k": k_values, "collocation_x": collocation,
        "bump_u": bump_u, "sampled_residual": sampled_residual,
        "sampled_residual_rms": rms, "bump_l2_error": bump_l2,
        "continuous_residual_l2": continuous_rms,
        "nonattainment_n": n_values, "nonattainment_u": affine_u,
        "nonattainment_l2_error": affine_l2_error,
        "nonattainment_residual_l2": affine_residual_l2,
        "nonattainment_residual_loss": affine_residual_l2**2,
        "nonattainment_output_weight": 1.0 / np.tanh(1.0 / n_values),
    }
    metadata = {
        "note": "Explicit network sequences; no optimizer or training is involved.",
        "source_note": "Worked tanh illustrations of the mechanism, not examples quoted from arXiv:2606.04018.",
        "pde": "-u''=0 on (0,1)",
        "bump": {
            "boundary_conditions": "u(0)=u(1)=0", "exact_solution": "0",
            "network": "[tanh(k*(x-a))-tanh(k*(x-b))-(tanh(k*b)-tanh(k*a))]/2",
            "a": A, "b": B, "k": k_values.tolist(),
            "collocation_x": collocation.tolist(),
            "sampled_residual_rms": rms.tolist(), "l2_error": bump_l2.tolist(),
            "continuous_residual_l2": continuous_rms.tolist(),
            "l2_error_limit": float(limit_l2),
            "reported_residuals_floored_or_truncated": False,
        },
        "nonattainment": {
            "boundary_conditions": "u(0)=0, u(1)=1", "exact_solution": "x",
            "network": "tanh(x/n)/tanh(1/n)",
            "architecture": "Finite shallow tanh network, linear output, no linear skip connection.",
            "n": n_values.tolist(), "l2_error": affine_l2_error.tolist(),
            "continuous_residual_l2": affine_residual_l2.tolist(),
            "continuous_residual_loss": (affine_residual_l2**2).tolist(),
            "output_weight": (1.0 / np.tanh(1.0 / n_values)).tolist(),
            "n_to_fourth_times_residual_loss": scaled_affine_loss.tolist(),
        },
        "verification": {
            "quadrature": "Composite trapezoidal rule on 65537 equally spaced nodes.",
            "refinement_nodes": int(x_fine.size),
            "l2_grid_refinement_max_difference": float(refinement_difference),
            "bump_boundary_max_abs": boundary_bump,
            "nonattainment_boundary_max_abs": boundary_affine,
            "bump_second_derivative_relative_fd_error_max": max(derivative_errors),
            "nonattainment_second_derivative_relative_fd_error_max": max(affine_derivative_errors),
            "limit_check_k": limit_check_k, "limit_check_l2": limit_check_l2,
            "limit_check_l2_difference": float(abs(limit_check_l2 - limit_l2)),
        },
    }
    np.savez_compressed(results / "counterexample_data.npz", **data)
    (results / "counterexample_data.json").write_text(json.dumps(metadata, indent=2) + "\n")
    draw(data)
    print("Saved counterexample data and residual_counterexample.pdf/.png.")
    print(f"k=256: sampled residual RMS={rms[-1]:.6e}; L2 solution error={bump_l2[-1]:.6f}.")
    print(f"n=32: residual loss={affine_residual_l2[-1]**2:.6e}; output weight={1 / np.tanh(1 / 32):.6f}.")
    print(f"Validated boundaries, analytic derivatives, quadrature, and L2 limit {limit_l2:.6f}.")


if __name__ == "__main__":
    main()

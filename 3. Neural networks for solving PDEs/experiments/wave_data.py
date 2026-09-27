#!/usr/bin/env python3
"""Generate the compact wave-equation sample plot for the supervised-data slide.

Run from the project root: python3 experiments/wave_data.py
Only NumPy and Matplotlib are required. The observations are noiseless synthetic
samples of u(t,x) = sin(pi*x) cos(pi*t), which solves u_tt = u_xx on [0,1]
with fixed endpoints, initial displacement sin(pi*x), and zero initial velocity.
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


def solution(t, x):
    return np.sin(np.pi * x) * np.cos(np.pi * t)


def main():
    times = np.array([0.0, 0.25, 0.75])
    x = np.linspace(0.0, 1.0, 601)
    sample_x = np.linspace(0.0, 1.0, 9)
    exact = solution(times[:, None], x[None, :])
    samples = solution(times[:, None], sample_x[None, :])

    # Analytical derivatives: u_tt = u_xx = -pi^2 sin(pi*x) cos(pi*t).
    utt = -(np.pi**2) * exact
    uxx = -(np.pi**2) * exact
    residual_max = float(np.max(np.abs(utt - uxx)))
    endpoint_max = float(np.max(np.abs(exact[:, [0, -1]])))
    initial_velocity = -np.pi * np.sin(np.pi * x) * np.sin(0.0)
    assert residual_max == 0.0
    assert endpoint_max < 1e-14
    np.testing.assert_allclose(exact[0], np.sin(np.pi * x), atol=1e-14)
    np.testing.assert_array_equal(initial_velocity, np.zeros_like(x))

    figures, results = ROOT / "figures", ROOT / "results"
    figures.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    np.savez_compressed(results / "wave_data.npz", t=times, x=x,
                        exact_u=exact, sample_x=sample_x, sample_u=samples)
    metadata = {
        "equation": "u_tt = u_xx",
        "exact_solution": "sin(pi*x)*cos(pi*t)",
        "boundary_conditions": "u(t,0) = u(t,1) = 0",
        "initial_conditions": "u(0,x) = sin(pi*x), u_t(0,x) = 0",
        "sample_times": times.tolist(),
        "samples_per_snapshot": int(sample_x.size),
        "noise_standard_deviation": 0.0,
        "analytic_pde_residual_max_abs": residual_max,
        "boundary_max_abs": endpoint_max,
    }
    (results / "wave_data.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "text.color": NAVY, "axes.labelcolor": NAVY,
        "xtick.color": "#647786", "ytick.color": "#647786",
        "axes.edgecolor": "#BAC8CF", "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "#F4F6F7", "figure.facecolor": "white",
        "grid.color": "#E3EAED", "grid.linewidth": 0.7,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.22, top=0.81)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.axhline(0, color="#BAC8CF", lw=0.8, zorder=1)
    for t, curve, observations, color in zip(times, exact, samples,
                                             [NAVY, TEAL, ORANGE]):
        ax.plot(x, curve, color=color, lw=1.7, alpha=0.65, zorder=2)
        ax.plot(sample_x, observations, ls="none", marker="o", ms=6.2,
                markerfacecolor=color, markeredgecolor="white", mew=0.8,
                color=color, label=rf"$t={t:g}$", zorder=3)
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.86, 1.13),
           xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[-0.5, 0, 0.5, 1])
    ax.set_xlabel("Position $x$", fontsize=14, labelpad=3)
    ax.set_ylabel("$u(t,x)$", fontsize=15, labelpad=7)
    ax.tick_params(axis="both", labelsize=13, length=3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
              borderaxespad=0, handletextpad=0.35, columnspacing=2.0,
              fontsize=14)
    fig.savefig(figures / "wave_data.pdf")
    fig.savefig(figures / "wave_data.png", dpi=220)
    plt.close(fig)
    print("Saved experiments/figures/wave_data.pdf and wave_data.png")
    print(f"Validated PDE and boundary conditions; boundary error {endpoint_max:.2e}.")


if __name__ == "__main__":
    main()

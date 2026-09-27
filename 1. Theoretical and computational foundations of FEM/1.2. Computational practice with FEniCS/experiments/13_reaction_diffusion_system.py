"""
13_reaction_diffusion_system.py
===============================

Gray-Scott coupled nonlinear reaction-diffusion system.

    u_t - D_u Delta u = -u v^2 + F(1-u)
    v_t - D_v Delta v =  u v^2 - (F+k)v

with homogeneous Neumann boundary conditions.  This compact finite-difference
solver generates the slide figure. The accompanying deck shows how to write
the mixed finite-element residual in FEniCS.

The script writes:
    reaction_diffusion_system.png
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)


N = 150
STEPS = 5200
DT = 1.0
DU = 0.16
DV = 0.08
FEED = 0.060
KILL = 0.062


def laplacian_neumann(a):
    padded = np.pad(a, 1, mode="edge")
    return (
        padded[1:-1, 2:]
        + padded[1:-1, :-2]
        + padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        - 4.0 * a
    )


def main():
    rng = np.random.default_rng(7)
    u = np.ones((N, N), dtype=float)
    v = np.zeros((N, N), dtype=float)

    r = N // 12
    c = N // 2
    u[c - r : c + r, c - r : c + r] = 0.50
    v[c - r : c + r, c - r : c + r] = 0.25
    u += 0.015 * rng.standard_normal((N, N))
    v += 0.015 * rng.standard_normal((N, N))
    u = np.clip(u, 0.0, 1.2)
    v = np.clip(v, 0.0, 1.2)

    snapshots = {}
    snapshot_steps = {0, 900, 2300, STEPS}
    snapshots[0] = (u.copy(), v.copy())

    for step in range(1, STEPS + 1):
        uvv = u * v * v
        u += DT * (DU * laplacian_neumann(u) - uvv + FEED * (1.0 - u))
        v += DT * (DV * laplacian_neumann(v) + uvv - (FEED + KILL) * v)
        if step in snapshot_steps:
            snapshots[step] = (u.copy(), v.copy())

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.2), constrained_layout=True)
    ordered_steps = sorted(snapshots)

    for col, step in enumerate(ordered_steps):
        u_snap, v_snap = snapshots[step]
        ax_u = axes[0, col]
        im_u = ax_u.imshow(u_snap, cmap="viridis", origin="lower", vmin=0.2, vmax=1.0)
        ax_u.set_title(f"u, step {step}")
        ax_u.set_xticks([])
        ax_u.set_yticks([])

        ax_v = axes[1, col]
        im_v = ax_v.imshow(v_snap, cmap="magma", origin="lower", vmin=0.0, vmax=0.45)
        ax_v.set_title(f"v, step {step}")
        ax_v.set_xticks([])
        ax_v.set_yticks([])

    fig.colorbar(im_u, ax=axes[0, :], shrink=0.78, location="right", pad=0.01)
    fig.colorbar(im_v, ax=axes[1, :], shrink=0.78, location="right", pad=0.01)
    fig.suptitle("Coupled nonlinear reaction-diffusion system: Gray-Scott patterns", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES / "reaction_diffusion_system.png", dpi=230, bbox_inches="tight")
    plt.close(fig)
    print("Saved reaction_diffusion_system.png")


if __name__ == "__main__":
    main()

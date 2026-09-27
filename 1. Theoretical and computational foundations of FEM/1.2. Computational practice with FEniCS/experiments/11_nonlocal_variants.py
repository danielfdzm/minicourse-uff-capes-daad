"""
11_nonlocal_variants.py
=======================

Solve and visualize variants of the nonlocal parabolic PDE

    u_t - ( a(x) M(t) u_x )_x = h(x) 1_O(x),
    M(t) = int_0^1 u(x,t) dx,

on Omega=(0,1), O=(0.35,0.65), with y0 initial data and homogeneous
Dirichlet boundary conditions at x=0 and x=1.

The script writes:
    nonlocal_variants.png
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)


T = 0.20
NUM_STEPS = 120
DT = T / NUM_STEPS
NX = 140
O_LEFT = 0.35
O_RIGHT = 0.65


def y0_profile(x):
    return 0.55 * np.sin(np.pi * x[0]) + 0.18 * np.sin(2.0 * np.pi * x[0])


VARIANTS = [
    {
        "name": "uniform diffusion, mild source",
        "a": lambda x: 0.28 + 0.0 * x[0],
        "h": lambda x: 1.15 + 0.0 * x[0],
        "color": "#1565c0",
    },
    {
        "name": "oscillatory a(x), left-biased h(x)",
        "a": lambda x: 0.22 * (1.0 + 0.55 * np.sin(2.0 * np.pi * x[0])) + 0.04,
        "h": lambda x: 1.25 * (1.0 + 0.60 * np.cos(2.0 * np.pi * x[0])),
        "color": "#2e7d32",
    },
    {
        "name": "right-conducting layer, localized h(x)",
        "a": lambda x: 0.08 + 0.34 / (1.0 + np.exp(-45.0 * (x[0] - 0.58))),
        "h": lambda x: 2.0 * np.exp(-95.0 * (x[0] - 0.48) ** 2),
        "color": "#e65100",
    },
]


def global_integral(msh, form):
    local_value = fem.assemble_scalar(form)
    return msh.comm.allreduce(local_value, op=MPI.SUM)


def gather_solution(msh, V, function):
    coords_local = V.tabulate_dof_coordinates()[:, 0]
    vals_local = function.x.array.real[: coords_local.shape[0]]
    local = np.column_stack((coords_local, vals_local))
    gathered = msh.comm.gather(local, root=0)

    if msh.comm.rank != 0:
        return None, None

    data = np.vstack(gathered)
    data = data[np.argsort(data[:, 0])]
    _, unique = np.unique(np.round(data[:, 0], 13), return_index=True)
    data = data[np.sort(unique)]
    return data[:, 0], data[:, 1]


def solve_variant(variant):
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, NX)
    V = fem.functionspace(msh, ("Lagrange", 1))

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)

    a_fun = fem.Function(V, name="a")
    h_fun = fem.Function(V, name="h_on_O")
    a_fun.interpolate(variant["a"])

    def h_on_o(x):
        mask = np.logical_and(x[0] >= O_LEFT, x[0] <= O_RIGHT)
        return np.where(mask, variant["h"](x), 0.0)

    h_fun.interpolate(h_on_o)

    u_n = fem.Function(V, name="u")
    u_n.interpolate(y0_profile)
    u_n.x.scatter_forward()

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt = fem.Constant(msh, default_scalar_type(DT))
    M = fem.Constant(msh, default_scalar_type(0.0))

    a_form = u * v * ufl.dx + dt * M * a_fun * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = u_n * v * ufl.dx + dt * h_fun * v * ufl.dx

    problem = LinearProblem(
        a_form,
        L_form,
        bcs=[bc],
        petsc_options_prefix="nonlocal_variant_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    snapshot_steps = {0, 12, 30, 60, 120}
    snapshots = []
    masses = []
    times = []

    x_vals, u_vals = gather_solution(msh, V, u_n)
    if msh.comm.rank == 0:
        snapshots.append((0.0, x_vals, u_vals.copy()))

    mass_form = fem.form(u_n * ufl.dx)
    masses.append(global_integral(msh, mass_form))
    times.append(0.0)

    for step in range(1, NUM_STEPS + 1):
        mass_k = global_integral(msh, mass_form)
        u_picard = fem.Function(V)
        u_picard.x.array[:] = u_n.x.array
        u_picard.x.scatter_forward()

        for _ in range(24):
            M.value = default_scalar_type(max(mass_k, 1.0e-9))
            u_new = problem.solve()
            u_new.x.scatter_forward()
            mass_new = global_integral(msh, fem.form(u_new * ufl.dx))
            update = msh.comm.allreduce(
                np.max(np.abs(u_new.x.array - u_picard.x.array)),
                op=MPI.MAX,
            )
            if update < 1.0e-9 and abs(mass_new - mass_k) < 1.0e-10:
                break
            u_picard.x.array[:] = u_new.x.array
            u_picard.x.scatter_forward()
            mass_k = mass_new

        u_n.x.array[:] = u_new.x.array
        u_n.x.scatter_forward()

        t = step * DT
        masses.append(global_integral(msh, mass_form))
        times.append(t)

        if step in snapshot_steps:
            x_vals, u_vals = gather_solution(msh, V, u_n)
            if msh.comm.rank == 0:
                snapshots.append((t, x_vals, u_vals.copy()))

    return {
        "name": variant["name"],
        "color": variant["color"],
        "snapshots": snapshots if msh.comm.rank == 0 else None,
        "times": np.asarray(times),
        "masses": np.asarray(masses),
    }


def make_figure(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.4), constrained_layout=True)
    snapshot_axes = axes.flat[:3]
    mass_ax = axes.flat[3]

    for ax, result in zip(snapshot_axes, results):
        for t, x_vals, u_vals in result["snapshots"]:
            alpha = 0.45 if t < T else 1.0
            width = 1.2 if t < T else 2.5
            ax.plot(x_vals, u_vals, linewidth=width, alpha=alpha, label=f"t={t:.3f}")
        ax.axvspan(O_LEFT, O_RIGHT, color="#f3e5ab", alpha=0.35, label="O" if ax is snapshot_axes[0] else None)
        ax.set_title(result["name"])
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=-0.02)

    snapshot_axes[0].legend(loc="upper right", fontsize=7, ncol=2)

    for result in results:
        mass_ax.plot(result["times"], result["masses"], color=result["color"], linewidth=2.2, label=result["name"])
    mass_ax.set_title(r"Nonlocal mass $M(t)=\int_0^1 u(x,t)\,dx$")
    mass_ax.set_xlabel("t")
    mass_ax.set_ylabel("M(t)")
    mass_ax.grid(True, alpha=0.25)
    mass_ax.legend(fontsize=7)

    fig.suptitle("Nonlocal parabolic equation: effect of a(x) and h(x)", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES / "nonlocal_variants.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    results = [solve_variant(variant) for variant in VARIANTS]
    if MPI.COMM_WORLD.rank == 0:
        make_figure(results)
        print("Saved nonlocal_variants.png")


if __name__ == "__main__":
    main()

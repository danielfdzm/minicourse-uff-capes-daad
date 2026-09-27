"""
12_heat_drift_control.py
========================

Forward-backward optimal control demo for a heat equation with drift.

State:
    y_t - nu y_xx + beta y_x = f

Cost:
    J(f) = 1/2 ||y(T)-y_T||^2 + alpha/2 int_0^T ||f||^2 dt

Adjoint:
    -p_t - nu p_xx - beta p_x = 0,   p(T)=y(T)-y_T

Optimality:
    alpha f + p = 0

The script writes:
    heat_drift_control.png
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


NX = 120
T = 0.35
NUM_STEPS = 110
DT = T / NUM_STEPS
NU = 0.015
BETA = 0.75
ALPHA = 0.12
CONTROL_ITERS = 24
GRADIENT_STEP = 0.45


def gaussian(center, width, amplitude=1.0):
    def profile(x):
        return amplitude * np.exp(-((x[0] - center) ** 2) / width)

    return profile


def build_problem():
    msh = mesh.create_unit_interval(MPI.COMM_WORLD, NX)
    V = fem.functionspace(msh, ("Lagrange", 1))

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)

    y0 = fem.Function(V, name="y0")
    y0.interpolate(gaussian(0.18, 0.006, 0.85))

    target = fem.Function(V, name="target")
    target.interpolate(gaussian(0.74, 0.010, 0.95))

    return msh, V, bc, boundary_dofs, y0, target


def clone_function(V, source=None, name="function"):
    out = fem.Function(V, name=name)
    if source is not None:
        out.x.array[:] = source.x.array
        out.x.scatter_forward()
    return out


def state_solver(msh, V, bc):
    y_n = fem.Function(V, name="y_previous")
    f_n = fem.Function(V, name="control")
    y = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt = fem.Constant(msh, default_scalar_type(DT))
    nu = fem.Constant(msh, default_scalar_type(NU))
    beta = fem.Constant(msh, default_scalar_type(BETA))

    a = y * v * ufl.dx + dt * nu * ufl.dot(ufl.grad(y), ufl.grad(v)) * ufl.dx
    a += dt * beta * ufl.grad(y)[0] * v * ufl.dx
    L = y_n * v * ufl.dx + dt * f_n * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="heat_state_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return problem, y_n, f_n


def adjoint_solver(msh, V, bc):
    p_next = fem.Function(V, name="p_next")
    p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    dt = fem.Constant(msh, default_scalar_type(DT))
    nu = fem.Constant(msh, default_scalar_type(NU))
    beta = fem.Constant(msh, default_scalar_type(BETA))

    a = p * q * ufl.dx + dt * nu * ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
    a -= dt * beta * ufl.grad(p)[0] * q * ufl.dx
    L = p_next * q * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="heat_adjoint_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return problem, p_next


def solve_state(msh, V, bc, y0, controls):
    problem, y_n, f_n = state_solver(msh, V, bc)
    y_n.x.array[:] = y0.x.array
    y_n.x.scatter_forward()

    states = [clone_function(V, y_n, "y_0")]
    for n, control in enumerate(controls):
        f_n.x.array[:] = control.x.array
        f_n.x.scatter_forward()
        y_new = problem.solve()
        y_new.x.scatter_forward()
        states.append(clone_function(V, y_new, f"y_{n + 1}"))
        y_n.x.array[:] = y_new.x.array
        y_n.x.scatter_forward()
    return states


def solve_adjoint(msh, V, bc, boundary_dofs, states, target):
    problem, p_next = adjoint_solver(msh, V, bc)
    p_next.x.array[:] = states[-1].x.array - target.x.array
    p_next.x.array[boundary_dofs] = 0.0
    p_next.x.scatter_forward()

    adjoints = [None] * (NUM_STEPS + 1)
    adjoints[-1] = clone_function(V, p_next, "p_T")

    for n in range(NUM_STEPS - 1, -1, -1):
        p_new = problem.solve()
        p_new.x.scatter_forward()
        adjoints[n] = clone_function(V, p_new, f"p_{n}")
        p_next.x.array[:] = p_new.x.array
        p_next.x.scatter_forward()
    return adjoints


def l2_error(msh, state, target):
    err_form = fem.form((state - target) ** 2 * ufl.dx)
    local = fem.assemble_scalar(err_form)
    return float(np.sqrt(msh.comm.allreduce(local, op=MPI.SUM)))


def control_norm(msh, controls):
    total = 0.0
    for control in controls:
        form = fem.form(control**2 * ufl.dx)
        total += DT * msh.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
    return float(np.sqrt(total))


def gather_series(msh, V, functions):
    coords_local = V.tabulate_dof_coordinates()[:, 0]
    order_local = np.argsort(coords_local)
    coords_local = coords_local[order_local]
    values_local = [fn.x.array.real[: coords_local.shape[0]][order_local] for fn in functions]
    local = np.column_stack([coords_local] + values_local)
    gathered = msh.comm.gather(local, root=0)

    if msh.comm.rank != 0:
        return None, None

    data = np.vstack(gathered)
    data = data[np.argsort(data[:, 0])]
    _, unique = np.unique(np.round(data[:, 0], 13), return_index=True)
    data = data[np.sort(unique)]
    return data[:, 0], data[:, 1:]


def make_figure(msh, V, y0, target, uncontrolled, controlled, controls, costs):
    import matplotlib.pyplot as plt

    x_vals, profiles = gather_series(msh, V, [y0, target, uncontrolled[-1], controlled[-1]])
    _, control_values = gather_series(msh, V, controls)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2), constrained_layout=True)

    ax0 = axes[0, 0]
    labels = ["initial", "target", "uncontrolled final", "controlled final"]
    colors = ["#757575", "#2e7d32", "#e65100", "#1565c0"]
    for j, (label, color) in enumerate(zip(labels, colors)):
        ax0.plot(x_vals, profiles[:, j], label=label, color=color, linewidth=2.2)
    ax0.set_title("State steering under drift")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    ax1 = axes[0, 1]
    time_grid = np.linspace(DT, T, NUM_STEPS)
    im = ax1.imshow(
        control_values.T,
        origin="lower",
        aspect="auto",
        extent=[x_vals[0], x_vals[-1], time_grid[0], time_grid[-1]],
        cmap="coolwarm",
    )
    ax1.set_title("Optimal source f(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    fig.colorbar(im, ax=ax1, shrink=0.82)

    ax2 = axes[1, 0]
    ax2.plot(np.arange(len(costs)), costs, marker="o", color="#7b1fa2", linewidth=2.0)
    ax2.set_title("Forward-backward sweep convergence")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("objective proxy")
    ax2.grid(True, alpha=0.25)

    ax3 = axes[1, 1]
    selected = [0, NUM_STEPS // 3, 2 * NUM_STEPS // 3, NUM_STEPS]
    _, state_values = gather_series(msh, V, [controlled[idx] for idx in selected])
    for j, idx in enumerate(selected):
        ax3.plot(x_vals, state_values[:, j], linewidth=2.0, label=f"t={idx * DT:.2f}")
    ax3.set_title("Controlled state snapshots")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8)

    fig.suptitle("Optimal control of a heat equation with drift", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES / "heat_drift_control.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    msh, V, bc, boundary_dofs, y0, target = build_problem()
    zero_controls = [fem.Function(V, name=f"zero_control_{n}") for n in range(NUM_STEPS)]
    uncontrolled = solve_state(msh, V, bc, y0, zero_controls)

    controls = [fem.Function(V, name=f"control_{n}") for n in range(NUM_STEPS)]
    costs = []

    for it in range(CONTROL_ITERS):
        states = solve_state(msh, V, bc, y0, controls)
        adjoints = solve_adjoint(msh, V, bc, boundary_dofs, states, target)

        terminal_error = l2_error(msh, states[-1], target)
        cnorm = control_norm(msh, controls)
        costs.append(0.5 * terminal_error**2 + 0.5 * ALPHA * cnorm**2)

        for n, control in enumerate(controls):
            # Reduced gradient for this sign convention: g = alpha*f + p.
            # A modest fixed step is more stable than imposing f=-p/alpha
            # in one forward-backward sweep.
            gradient = ALPHA * control.x.array + adjoints[n + 1].x.array
            control.x.array[:] = control.x.array - GRADIENT_STEP * gradient
            control.x.array[:] = np.clip(control.x.array, -8.0, 8.0)
            control.x.scatter_forward()

        if msh.comm.rank == 0:
            print(f"iter={it:02d}, terminal L2={terminal_error:.4e}, ||f||={cnorm:.4e}")

    controlled = solve_state(msh, V, bc, y0, controls)

    if msh.comm.rank == 0:
        make_figure(msh, V, y0, target, uncontrolled, controlled, controls, costs)
        print("Saved heat_drift_control.png")


if __name__ == "__main__":
    main()

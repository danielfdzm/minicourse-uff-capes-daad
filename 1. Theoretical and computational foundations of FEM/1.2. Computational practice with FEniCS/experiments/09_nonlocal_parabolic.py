r"""
09_nonlocal_parabolic.py
=======================

Solve the nonlocal PDE on Omega=(0,1):

    u_t + ( a(x) * M(t) * u_x )_x = 0,
    M(t) = \int_0^1 u(x,t) dx,

with homogeneous Dirichlet boundary conditions and initial data u(x,0)=y0(x).

Time discretization: backward Euler.
Nonlinearity handling: Picard iteration at each time step for M(t).
The time history is written to XDMF and a snapshot/mass plot is saved as
nonlocal_parabolic_solution.png.

The weak form below follows the sign exactly as written above:

    (u^{n+1}, v) - dt*M(u^{n+1})*(a*u_x^{n+1}, v_x) = (u^n, v).

For positive a(x) and positive M(t), this has the backward-heat sign. If your
intended model is the usual diffusive equation

    u_t - (a(x) M(t) u_x)_x = 0,

change the minus sign in a_form to a plus sign.
"""

import numpy as np
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
import ufl
from pathlib import Path

# ── Output folders (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# ---- User parameters ----
# The PDE sign written above is anti-diffusive when a(x)M(t) > 0, so the
# demo uses a small final time and time step. For the diffusive sign noted in
# the docstring, much larger time steps are typically fine.
T = 1.0e-3
num_steps = 500
dt = T / num_steps

nx = 100

picard_mass_tol = 1.0e-10
picard_update_tol = 1.0e-8
picard_max_it = 30


# ---- Mesh and function space ----
msh = mesh.create_unit_interval(MPI.COMM_WORLD, nx)
V = fem.functionspace(msh, ("Lagrange", 1))


# ---- Homogeneous Dirichlet boundary condition on x=0 and x=1 ----
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), boundary_dofs, V)


# ---- Coefficient a(x) and initial condition y0(x) ----
a_fun = fem.Function(V, name="a")


def a_profile(x):
    return 1.0 + 0.30 * np.sin(2.0 * np.pi * x[0])


def y0_profile(x):
    return np.sin(np.pi * x[0])


a_fun.interpolate(a_profile)

u_n = fem.Function(V, name="u")
u_n.interpolate(y0_profile)
u_n.x.scatter_forward()


# ---- Helper: global integral over Omega ----
def global_integral(form):
    local_val = fem.assemble_scalar(form)
    return msh.comm.allreduce(local_val, op=MPI.SUM)


mass_form_u_n = fem.form(u_n * ufl.dx)


# ---- Plot helpers ----
snapshot_steps = set(np.linspace(0, num_steps, 6, dtype=int))
snapshots = []
times = []
mass_values = []


def gather_solution(function):
    x_local = V.tabulate_dof_coordinates()[:, 0]
    u_local = function.x.array.real[: len(x_local)]
    local_data = np.column_stack((x_local, u_local))
    gathered = msh.comm.gather(local_data, root=0)

    if msh.comm.rank != 0:
        return None, None

    data = np.vstack(gathered)
    data = data[np.argsort(data[:, 0])]
    _, unique_indices = np.unique(np.round(data[:, 0], 14), return_index=True)
    data = data[np.sort(unique_indices)]
    return data[:, 0], data[:, 1]


def record_snapshot(function, time):
    x_vals, u_vals = gather_solution(function)
    if msh.comm.rank == 0:
        snapshots.append((time, x_vals, u_vals.copy()))


initial_mass = global_integral(mass_form_u_n)
if msh.comm.rank == 0:
    times.append(0.0)
    mass_values.append(initial_mass)
record_snapshot(u_n, 0.0)


# ---- Variational forms (backward Euler + Picard for nonlocal mass) ----
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dt_const = fem.Constant(msh, default_scalar_type(dt))
M_const = fem.Constant(msh, default_scalar_type(0.0))

# Weak form at each Picard iteration:
# (u^{n+1}, v) - dt * M_k * (a(x) u_x^{n+1}, v_x) = (u^n, v)
a_form = (
    u * v * ufl.dx
    - dt_const * M_const * a_fun * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
)
L_form = u_n * v * ufl.dx

problem = LinearProblem(
    a_form,
    L_form,
    bcs=[bc],
    petsc_options_prefix="nonlocal_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)


# ---- Time stepping ----
t = 0.0

if msh.comm.rank == 0:
    print("Starting nonlocal solve...")
    print(f"T={T}, steps={num_steps}, dt={dt:.3e}, nx={nx}")

with XDMFFile(msh.comm, str(RESULTS / "nonlocal_parabolic.xdmf"), "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_n, t)

    for step in range(1, num_steps + 1):
        t += dt

        # Picard initial guess from previous time level mass
        mass_k = global_integral(mass_form_u_n)
        u_prev_picard = fem.Function(V)
        u_prev_picard.x.array[:] = u_n.x.array
        u_prev_picard.x.scatter_forward()

        converged = False
        for k in range(1, picard_max_it + 1):
            M_const.value = default_scalar_type(mass_k)
            u_new = problem.solve()
            u_new.x.scatter_forward()

            mass_new = global_integral(fem.form(u_new * ufl.dx))
            local_diff = np.max(np.abs(u_new.x.array - u_prev_picard.x.array))
            inf_diff = msh.comm.allreduce(local_diff, op=MPI.MAX)

            rel_mass_change = abs(mass_new - mass_k) / max(1.0, abs(mass_new))

            if inf_diff < picard_update_tol and rel_mass_change < picard_mass_tol:
                converged = True
                break

            u_prev_picard.x.array[:] = u_new.x.array
            u_prev_picard.x.scatter_forward()
            mass_k = mass_new

        if not converged:
            raise RuntimeError(
                f"Picard iteration failed at step {step}, t={t:.6f}. "
                f"Last inf-diff={inf_diff:.3e}, rel-mass-change={rel_mass_change:.3e}"
            )

        u_n.x.array[:] = u_new.x.array
        u_n.x.scatter_forward()
        xdmf.write_function(u_n, t)

        mass_val = global_integral(mass_form_u_n)
        if msh.comm.rank == 0:
            times.append(t)
            mass_values.append(mass_val)

        if step in snapshot_steps:
            record_snapshot(u_n, t)

        if msh.comm.rank == 0 and (step % 25 == 0 or step == 1):
            print(
                f"step {step:4d}/{num_steps}, t={t:.6e}, "
                f"Picard it={k:2d}, mass={mass_val:.6e}"
            )

if msh.comm.rank == 0:
    print("Done. Solution written to nonlocal_parabolic.xdmf")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    for time, x_vals, u_vals in snapshots:
        axes[0].plot(x_vals, u_vals, linewidth=2.0, label=f"t={time:.1e}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x,t)")
    axes[0].set_title("Solution snapshots")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(times, mass_values, color="#0f766e", linewidth=2.2)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$\int_0^1 u(x,t)\,dx$")
    axes[1].set_title("Nonlocal coefficient M(t)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Nonlocal parabolic equation", fontsize=18, fontweight="bold")
    fig.savefig(FIGURES / "nonlocal_parabolic_solution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Plot saved to nonlocal_parabolic_solution.png")

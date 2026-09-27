"""
03_heat_equation.py — Time-dependent heat equation with FEniCSx
===============================================================

Problem:
    du/dt - Laplacian(u) = 0   in  Omega x (0, T]
                       u = 0   on  dOmega x (0, T]
                    u(0) = Gaussian bump

Time discretization: Backward Euler
    (u^{n+1} - u^n)/dt - Lap(u^{n+1}) = 0

Weak form at each time step:
    (u, v) + dt*(grad(u), grad(v)) = (u_n, v)
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import matplotlib.pyplot as plt
from pathlib import Path

# ── Output folders (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── Parameters ──
T = 0.25           # Final time (shorter window to better see diffusion dynamics)
num_steps = 100    # Number of time steps
dt = T / num_steps

# ── Mesh and function space ──
nx = ny = 30
msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
V = fem.functionspace(msh, ("Lagrange", 1))

# ── Boundary condition: u = 0 on dOmega ──
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(
    default_scalar_type(0.0),
    boundary_dofs, V)

# ── Initial condition: Gaussian bump ──
def gaussian_bump(x):
    return np.exp(-25 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

u_n = fem.Function(V, name="u_n")
u_n.interpolate(gaussian_bump)

# Diagnostics over time
mass_form = fem.form(u_n * ufl.dx)
energy_form = fem.form(u_n * u_n * ufl.dx)
times = [0.0]
peak_vals = [float(np.max(u_n.x.array.real))]
mass_vals = [msh.comm.allreduce(fem.assemble_scalar(mass_form), op=MPI.SUM)]
energy_vals = [msh.comm.allreduce(fem.assemble_scalar(energy_form), op=MPI.SUM)]

# ── Variational problem (backward Euler) ──
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

dt_const = fem.Constant(msh, default_scalar_type(dt))

a = u * v * ufl.dx + dt_const * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = u_n * v * ufl.dx

# ── Time stepping ──
# Store snapshots for plotting
snapshots = []
snap_times = [0.0, 0.02, 0.05, 0.10, 0.20]
snap_idx = 0

# Save initial condition
if snap_idx < len(snap_times) and abs(0.0 - snap_times[snap_idx]) < 1e-10:
    snap = fem.Function(V)
    snap.x.array[:] = u_n.x.array
    snapshots.append((0.0, snap))
    snap_idx += 1

u_h = fem.Function(V, name="u_h")
t = 0.0

for n in range(num_steps):
    t += dt

    # Solve the linear system
    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options_prefix="heat_",
                            petsc_options={"ksp_type": "preonly",
                                           "pc_type": "lu"})
    u_h = problem.solve()

    # Save snapshot if needed
    if snap_idx < len(snap_times) and abs(t - snap_times[snap_idx]) < dt / 2:
        snap = fem.Function(V)
        snap.x.array[:] = u_h.x.array
        snapshots.append((t, snap))
        snap_idx += 1

    # Update previous solution: u_n <- u^{n+1}
    u_n.x.array[:] = u_h.x.array

    times.append(t)
    peak_vals.append(float(np.max(u_n.x.array.real)))
    mass_vals.append(msh.comm.allreduce(fem.assemble_scalar(mass_form), op=MPI.SUM))
    energy_vals.append(msh.comm.allreduce(fem.assemble_scalar(energy_form), op=MPI.SUM))

    if (n + 1) % 20 == 0:
        print(f"  Step {n+1}/{num_steps}, t = {t:.3f}")

# ── Plot snapshots ──
from dolfinx.plot import vtk_mesh
import matplotlib.tri as mtri

topology, _, geometry = vtk_mesh(V)
triangles = topology.reshape((-1, 4))[:, 1:4]
triang = mtri.Triangulation(geometry[:, 0], geometry[:, 1], triangles)

ncols = len(snapshots)
fig_snap, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 5.3),
                              constrained_layout=False,
                              sharex=True, sharey=True)
if ncols == 1:
    axes = [axes]

u_max = max(float(np.max(s.x.array.real)) for _, s in snapshots)
levels = np.linspace(0.0, u_max, 30)
last_cf = None

for i, (time, snap) in enumerate(snapshots):
    values = snap.x.array.real
    last_cf = axes[i].tricontourf(triang, values, levels=levels,
                                  cmap="magma", vmin=0.0, vmax=u_max)
    axes[i].tricontour(triang, values, levels=10, colors="white",
                       linewidths=0.8, alpha=0.6)
    axes[i].set_aspect("equal", adjustable="box")
    axes[i].set_title(f"t = {time:.3f}", fontsize=30, fontweight="bold")
    axes[i].set_xlabel("x", fontsize=16)
    if i == 0:
        axes[i].set_ylabel("y", fontsize=16)
    axes[i].tick_params(axis="both", labelsize=13)

cbar = fig_snap.colorbar(last_cf, ax=axes, location="right", shrink=0.9, pad=0.015)
cbar.set_label("u(x, y, t)", fontsize=17)
cbar.ax.tick_params(labelsize=13)
# Reserve extra headroom so the global title does not intersect subplot titles.
fig_snap.subplots_adjust(top=0.80, bottom=0.12, left=0.05, right=0.92, wspace=0.20)
fig_snap.suptitle("Heat Equation Snapshots (early-time diffusion)",
                  fontsize=38, fontweight="bold", y=0.98)
fig_snap.savefig(FIGURES / "heat_equation.png", dpi=320, bbox_inches="tight", transparent=True)
plt.close(fig_snap)
print("Snapshots saved to heat_equation.png")

# Additional diagnostics plot
fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.6), constrained_layout=True)

ax[0].plot(times, peak_vals, color="#b91c1c", linewidth=2.4)
ax[0].fill_between(times, peak_vals, alpha=0.18, color="#f97316")
ax[0].set_xlabel("time", fontsize=14)
ax[0].set_ylabel("max(u)", fontsize=14)
ax[0].set_title("Peak temperature decay", fontsize=28, fontweight="bold")
ax[0].tick_params(labelsize=12)
ax[0].grid(True, alpha=0.3)

ax[1].semilogy(times, energy_vals, color="#1d4ed8", linewidth=2.4, label="$\\int u^2 dx$")
ax[1].semilogy(times, mass_vals, color="#0f766e", linewidth=2.0, linestyle="--", label="$\\int u dx$")
ax[1].set_xlabel("time", fontsize=14)
ax[1].set_ylabel("global quantity", fontsize=14)
ax[1].set_title("Dissipation diagnostics", fontsize=28, fontweight="bold")
ax[1].tick_params(labelsize=12)
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=12)

fig.savefig(FIGURES / "heat_decay_metrics.png", dpi=320, bbox_inches='tight', transparent=True)
plt.close(fig)
print("Diagnostics saved to heat_decay_metrics.png")

# Also export final solution to XDMF
from dolfinx.io import XDMFFile
with XDMFFile(msh.comm, str(RESULTS / "heat_final.xdmf"), "w") as xdmf:
    xdmf.write_mesh(msh)
    u_h.name = "u"
    xdmf.write_function(u_h)
    print("Final solution exported to heat_final.xdmf")

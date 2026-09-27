"""
02_convergence.py — Convergence study for the Poisson equation (FEniCSx)
========================================================================

Solves -Lap(u) = f on [0,1]^2 with known exact solution
on successively refined meshes and plots error vs h
to verify optimal convergence rates.
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 32,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

# Exact solution
def u_exact_func(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

def compute_errors(N, degree):
    """Solve Poisson on an NxN mesh with Pk elements, return (h, L2, H1)."""
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])

    # Interpolate exact solution for BC and error computation
    u_D = fem.Function(V)
    u_D.interpolate(u_exact_func)

    # Boundary condition
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_D, boundary_dofs)

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = 2 * np.pi**2 * u_exact

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # Solve
    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options_prefix=f"conv_p{degree}_n{N}_",
                            petsc_options={"ksp_type": "preonly",
                                           "pc_type": "lu"})
    u_h = problem.solve()

    # Compute errors
    # L2 error
    e = u_h - u_exact
    L2_form = fem.form(ufl.inner(e, e) * ufl.dx)
    E_L2 = np.sqrt(msh.comm.allreduce(
        fem.assemble_scalar(L2_form), op=MPI.SUM))

    # H1 seminorm error
    grad_e = ufl.grad(e)
    H1_form = fem.form(ufl.inner(grad_e, grad_e) * ufl.dx)
    E_H1_semi = np.sqrt(msh.comm.allreduce(
        fem.assemble_scalar(H1_form), op=MPI.SUM))

    # Full H1 error
    E_H1 = np.sqrt(E_L2**2 + E_H1_semi**2)

    # Mesh size h
    h = 1.0 / N  # For uniform mesh on unit square

    return h, E_L2, E_H1


# ── Convergence study: P1 and P2 elements ──
results = {}

for degree in [1, 2]:
    h_list, err_L2, err_H1 = [], [], []

    for N in [4, 8, 16, 32, 64, 128]:
        h, e_L2, e_H1 = compute_errors(N, degree)
        h_list.append(h)
        err_L2.append(e_L2)
        err_H1.append(e_H1)

        print(f"P{degree}: N={N:4d}, h={h:.4f}, "
              f"L2={e_L2:.4e}, H1={e_H1:.4e}")

    results[degree] = {
        'h': np.array(h_list),
        'L2': np.array(err_L2),
        'H1': np.array(err_H1),
    }

# ── Compute convergence rates ──
print("\nConvergence rates:")
rate_summary = {}
for degree, data in results.items():
    h, L2, H1 = data['h'], data['L2'], data['H1']
    rates_L2 = np.log(L2[:-1] / L2[1:]) / np.log(h[:-1] / h[1:])
    rates_H1 = np.log(H1[:-1] / H1[1:]) / np.log(h[:-1] / h[1:])
    rate_summary[degree] = {"L2": rates_L2, "H1": rates_H1}
    print(f"  P{degree} L2 rates: {rates_L2}")
    print(f"  P{degree} H1 rates: {rates_H1}")

# ── Plot convergence split by polynomial degree (slide-friendly) ──
colors = {1: '#1f77b4', 2: '#d62728'}
for degree in [1, 2]:
    data = results[degree]
    h, L2, H1 = data['h'], data['L2'], data['H1']

    fig, ax = plt.subplots(1, 1, figsize=(13.2, 9.4), constrained_layout=True)

    ax.loglog(h, L2, 'o-', color=colors[degree],
              label='$L^2$ error', linewidth=3.2, markersize=10,
              markeredgecolor='black', markeredgewidth=0.8)
    ax.loglog(h, H1, 's--', color=colors[degree],
              label='$H^1$ error', linewidth=3.0, markersize=9,
              markeredgecolor='black', markeredgewidth=0.8)

    # Reference slopes anchored at the coarsest point for this degree.
    order = np.argsort(h)[::-1]
    h_ref = h[order]
    L2_ref_data = L2[order]
    H1_ref_data = H1[order]
    h0 = h_ref[0]
    ref_l2 = L2_ref_data[0] * (h_ref / h0)**(degree + 1)
    ref_h1 = H1_ref_data[0] * (h_ref / h0)**degree

    ax.loglog(h_ref, ref_l2, ':', color=colors[degree], linewidth=2.3, alpha=0.7,
              label=fr'Ref $L^2$: $O(h^{degree+1})$')
    ax.loglog(h_ref, ref_h1, '--', color=colors[degree], linewidth=2.1, alpha=0.6,
              label=fr'Ref $H^1$: $O(h^{degree})$')

    ax.set_xlabel('$h$ (mesh size)')
    ax.set_ylabel('Error')
    ax.set_title(fr'Convergence for $P_{degree}$ elements', fontsize=34, fontweight='bold')
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(loc='best', frameon=True, fontsize=18)

    fig.savefig(FIGURES / f"convergence_plot_p{degree}.png", dpi=320,
                bbox_inches='tight', transparent=True)
    plt.close(fig)

# ── Plot observed rates per refinement level ──
levels = np.arange(1, len(results[1]['h']))
fig_rates, ax_rates = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)

for degree in [1, 2]:
    ax_rates[0].plot(levels, rate_summary[degree]['L2'], 'o-', linewidth=2.2,
                     markersize=6, color=colors[degree], label=f'$P_{degree}$')
    ax_rates[1].plot(levels, rate_summary[degree]['H1'], 's-', linewidth=2.2,
                     markersize=6, color=colors[degree], label=f'$P_{degree}$')

ax_rates[0].axhline(2, color=colors[1], linestyle='--', alpha=0.55)
ax_rates[0].axhline(3, color=colors[2], linestyle='--', alpha=0.55)
ax_rates[1].axhline(1, color=colors[1], linestyle='--', alpha=0.55)
ax_rates[1].axhline(2, color=colors[2], linestyle='--', alpha=0.55)

ax_rates[0].set_title('Observed $L^2$ rates', fontsize=28, fontweight='bold')
ax_rates[1].set_title('Observed $H^1$ rates', fontsize=28, fontweight='bold')
for ax in ax_rates:
    ax.set_xlabel('Refinement level', fontsize=15)
    ax.set_ylabel('Rate', fontsize=15)
    ax.set_xticks(levels)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=11)

fig_rates.savefig(FIGURES / "convergence_rates.png", dpi=320, bbox_inches='tight', transparent=True)
plt.close(fig_rates)

# ── L2 convergence plotted against mesh size and polynomial degree ──
fig3d = plt.figure(figsize=(11.8, 8.6))
ax3d = fig3d.add_subplot(111, projection='3d')

for degree in [1, 2]:
    x = np.log10(1.0 / results[degree]['h'])
    y = np.full_like(x, degree, dtype=float)
    z = np.log10(results[degree]['L2'])
    ax3d.plot(x, y, z, '-o', linewidth=2.6, markersize=7,
              color=colors[degree], label=f'$P_{degree}$')
    ax3d.scatter(x, y, z, s=46, color=colors[degree], edgecolors='black')

ax3d.set_xlabel('$\\log_{10}(1/h)$', fontsize=15)
ax3d.set_ylabel('Polynomial degree', fontsize=15)
ax3d.set_zlabel('$\\log_{10}(L^2\\ error)$', fontsize=15)
ax3d.set_title('3D Convergence Trajectories', fontsize=32, fontweight='bold')
ax3d.view_init(elev=24, azim=-57)
ax3d.legend(fontsize=11)

fig3d.savefig(FIGURES / "convergence_3d.png", dpi=320, bbox_inches='tight', transparent=True)
plt.close(fig3d)

print("\nSaved: convergence_plot_p1.png, convergence_plot_p2.png, "
    "convergence_rates.png, convergence_3d.png")

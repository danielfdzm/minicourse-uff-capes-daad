"""
04_nonlinear.py — Nonlinear Poisson equation with FEniCSx
=========================================================

Problem:
    -div( (1 + u^2) * grad(u) ) = f   in  Omega
                               u = 0   on  dOmega

FEniCSx solves this with PETSc SNES via NonlinearProblem.
You write the residual F(u; v) = 0 and the Jacobian
J = dF/du is computed automatically.
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import matplotlib.pyplot as plt
from pathlib import Path

# ── Output folders (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── Mesh and function space ──
msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.functionspace(msh, ("Lagrange", 1))

# ── Boundary condition ──
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(
    default_scalar_type(0.0),
    boundary_dofs, V)

# ── Source term ──
def f_func(x):
    return x[0] * np.sin(np.pi * x[1])

f = fem.Function(V)
f.interpolate(f_func)

# ── Define the nonlinear variational problem ──
# Use a Function for the unknown in a nonlinear problem
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)

# Nonlinear residual F(u; v) = 0
F = (1 + u**2) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    - f * v * ufl.dx

# Jacobian (computed automatically by UFL differentiation)
J = ufl.derivative(F, u)

# ── Create nonlinear problem and configure PETSc SNES ──
problem = NonlinearProblem(
    F, u, bcs=[bc], J=J,
    petsc_options_prefix="nonlinear_",
    petsc_options={
        "snes_type": "newtonls",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
        "ksp_type": "preonly",
        "pc_type": "lu",
    },
)

# ── Solve ──
log.set_log_level(log.LogLevel.INFO)
problem.solve()
num_iterations = problem.solver.getIterationNumber()
if problem.solver.getConvergedReason() <= 0:
    raise RuntimeError("Newton solver failed to converge")
print(f"\nNewton solver converged in {num_iterations} iterations")

# Derived coefficient for post-processing
W = fem.functionspace(msh, ("DG", 0))
kappa_expr = fem.Expression(1 + u**2, W.element.interpolation_points)
kappa = fem.Function(W, name="kappa")
kappa.interpolate(kappa_expr)

# ── Visualize ──
try:
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    pv.set_jupyter_backend("static")
    topology_u, cell_types_u, geometry_u = vtk_mesh(V)
    grid_u = pv.UnstructuredGrid(topology_u, cell_types_u, geometry_u)
    grid_u.point_data["u"] = u.x.array.real
    u_abs_max = float(np.max(np.abs(grid_u.point_data["u"])))
    warp_scale = 0.18 / max(u_abs_max, 1.0e-12)
    warped_u = grid_u.warp_by_scalar("u", factor=warp_scale)
    contours_u = grid_u.contour(isosurfaces=14, scalars="u")

    topology_k, cell_types_k, geometry_k = vtk_mesh(msh, tdim)
    grid_k = pv.UnstructuredGrid(topology_k, cell_types_k, geometry_k)
    grid_k.cell_data["kappa"] = kappa.x.array.real

    plotter = pv.Plotter(shape=(1, 3), window_size=(1900, 620))

    plotter.subplot(0, 0)
    plotter.add_mesh(grid_u, scalars="u", show_edges=False,
                     cmap="viridis",
                     scalar_bar_args={"title": "u", "fmt": "%.3f"})
    plotter.add_title("Solution field (2D)", font_size=38)
    plotter.view_xy()

    plotter.subplot(0, 1)
    plotter.add_mesh(warped_u, scalars="u", cmap="viridis",
                     smooth_shading=True, specular=0.3,
                     scalar_bar_args={"title": "u", "fmt": "%.3f"})
    plotter.add_mesh(contours_u, color="white", opacity=0.6, line_width=1.2)
    plotter.add_title("Warped solution (3D)", font_size=38)
    plotter.view_isometric()
    plotter.reset_camera()

    plotter.subplot(0, 2)
    plotter.add_mesh(grid_k, scalars="kappa", show_edges=False,
                     cmap="inferno",
                     scalar_bar_args={"title": "1 + u^2", "fmt": "%.3f"})
    plotter.add_title("Nonlinear coefficient kappa", font_size=38)
    plotter.view_xy()

    plotter.screenshot(str(FIGURES / "nonlinear_poisson.png"), scale=3, transparent_background=True)
    plotter.close()

    # Additional dedicated 3D shot
    plotter3d = pv.Plotter(window_size=(1300, 900))
    plotter3d.set_background("#f8fafc")
    plotter3d.add_mesh(warped_u, scalars="u", cmap="turbo",
                       smooth_shading=True, specular=0.35,
                       scalar_bar_args={"title": "u", "fmt": "%.3f"})
    plotter3d.add_mesh(contours_u, color="#0f172a", opacity=0.5, line_width=1.0)
    plotter3d.add_title("Nonlinear Poisson solution (3D)", font_size=40)
    plotter3d.view_isometric()
    plotter3d.reset_camera()
    plotter3d.camera.zoom(0.9)
    plotter3d.screenshot(str(FIGURES / "nonlinear_poisson_3d.png"), scale=3, transparent_background=True)
    plotter3d.close()

    print("Solutions saved to nonlinear_poisson.png and nonlinear_poisson_3d.png")

except ImportError:
    print("Install pyvista for visualization: pip install pyvista")

# Distribution of the nonlinear diffusion coefficient
fig, ax = plt.subplots(figsize=(7.8, 4.8))
ax.hist(kappa.x.array.real, bins=30, color="#f97316", edgecolor="#7c2d12", alpha=0.85)
ax.set_xlabel("kappa = 1 + u^2")
ax.set_ylabel("Cell count")
ax.set_title("Distribution of nonlinear diffusion coefficient", fontsize=30, fontweight="bold")
ax.grid(True, axis='y', alpha=0.25)
fig.savefig(FIGURES / "nonlinear_kappa_hist.png", dpi=260, bbox_inches='tight', transparent=True)
plt.close(fig)
print("Saved nonlinear_kappa_hist.png")

# Export to XDMF
from dolfinx.io import XDMFFile
with XDMFFile(msh.comm, str(RESULTS / "nonlinear_poisson.xdmf"), "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u)
    print("Solution exported to nonlinear_poisson.xdmf")

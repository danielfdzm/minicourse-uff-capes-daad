"""
01_poisson.py — Solving the Poisson equation with FEniCSx
=========================================================

Problem:
    -Laplacian(u) = f   in  Omega = [0,1]^2
              u   = u_D on  dOmega

with f = -6  and  u_D = 1 + x^2 + 2y^2
(so the exact solution is u = 1 + x^2 + 2y^2).
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
from pathlib import Path

# ── Output folders (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── 1. Create mesh and function space ──
msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
V = fem.functionspace(msh, ("Lagrange", 1))  # P1 elements

# ── 2. Define boundary condition ──
# Exact solution as a callable
def u_exact_func(x):
    return 1 + x[0]**2 + 2 * x[1]**2

u_D = fem.Function(V)
u_D.interpolate(u_exact_func)

# Find boundary DOFs
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_D, boundary_dofs)

# ── 3. Define variational problem ──
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(msh, default_scalar_type(-6.0))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx    # Bilinear form
L = f * v * ufl.dx                                   # Linear form

# ── 4. Solve ──
problem = LinearProblem(a, L, bcs=[bc],
                        petsc_options_prefix="poisson_",
                        petsc_options={"ksp_type": "preonly",
                                       "pc_type": "lu"})
u_h = problem.solve()
u_h.name = "u_h"

# ── 5. Compute error ──
# L2 error
error_L2 = fem.form(ufl.inner(u_h - u_D, u_h - u_D) * ufl.dx)
E_L2 = np.sqrt(msh.comm.allreduce(
    fem.assemble_scalar(error_L2), op=MPI.SUM))

# H1 semi-norm error
grad_err = ufl.grad(u_h - u_D)
error_H1 = fem.form(ufl.inner(grad_err, grad_err) * ufl.dx)
E_H1_semi = np.sqrt(msh.comm.allreduce(
    fem.assemble_scalar(error_H1), op=MPI.SUM))

# Full H1 error
E_H1 = np.sqrt(E_L2**2 + E_H1_semi**2)

print(f"L2 error:       {E_L2:.6e}")
print(f"H1 semi error:  {E_H1_semi:.6e}")
print(f"H1 error:       {E_H1:.6e}")

# ── 6. Plot solution ──
# Option A: Using pyvista (recommended for FEniCSx)
try:
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    pv.set_jupyter_backend("static")

    topology_m, cell_types_m, geometry_m = vtk_mesh(msh, tdim)
    mesh_grid = pv.UnstructuredGrid(topology_m, cell_types_m, geometry_m)

    topology_v, cell_types_v, geometry_v = vtk_mesh(V)
    solution_grid = pv.UnstructuredGrid(topology_v, cell_types_v, geometry_v)
    solution_grid.point_data["u_h"] = u_h.x.array.real
    u_max = float(np.max(solution_grid.point_data["u_h"]))
    scale = 0.18 / max(u_max, 1.0e-12)
    warped = solution_grid.warp_by_scalar("u_h", factor=scale)
    contours = solution_grid.contour(isosurfaces=14, scalars="u_h")

    scalar_bar_args = {
        "title": "u_h",
        "fmt": "%.3f",
        "title_font_size": 34,
        "label_font_size": 20,
        "vertical": True,
    }

    # Slide-friendly 2D figure: larger canvas for clearer side-by-side panels.
    plotter = pv.Plotter(shape=(1, 2), window_size=(3000, 1300))
    plotter.set_background("#f6f8fb")

    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_grid, show_edges=True, color="#f8f9fa",
                     edge_color="#111827", line_width=2.3)
    plotter.add_title("Finite-element mesh", font_size=40)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.08)

    plotter.subplot(0, 1)
    plotter.add_mesh(solution_grid, scalars="u_h", cmap="viridis",
                     show_edges=False,
                     scalar_bar_args=scalar_bar_args)
    plotter.add_mesh(contours, color="white", line_width=2.0, opacity=0.6)
    plotter.add_title("Poisson solution (2D)", font_size=40)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.03)

    plotter.screenshot(str(FIGURES / "poisson_solution.png"), scale=3, transparent_background=True)
    plotter.close()

    # Dedicated 3D render: keep a flat-lit surface (no light/shade gradients).
    plotter3d = pv.Plotter(window_size=(1820, 1260))
    plotter3d.set_background("#eef2f7", top="#ffffff")

    warped_contours = warped.contour(isosurfaces=14, scalars="u_h")

    plotter3d.add_mesh(warped, scalars="u_h", cmap="turbo",
                       smooth_shading=False,
                       lighting=False,
                       scalar_bar_args=scalar_bar_args)
    plotter3d.add_mesh(warped_contours, color="#111827", line_width=1.2, opacity=0.55)
    plotter3d.add_title("Poisson solution: 3D surface", font_size=42)
    plotter3d.camera_position = [
        (2.05, -1.45, 1.35),
        (0.50, 0.50, 0.16),
        (0.0, 0.0, 1.0),
    ]
    plotter3d.reset_camera()
    plotter3d.camera.zoom(1.05)
    plotter3d.screenshot(str(FIGURES / "poisson_solution_3d.png"), scale=3, transparent_background=True)
    plotter3d.close()
    print("Solutions saved to poisson_solution.png and poisson_solution_3d.png")

except ImportError:
    print("Install pyvista for visualization: pip install pyvista")

# Option B: Export to file for ParaView
from dolfinx.io import XDMFFile
with XDMFFile(msh.comm, str(RESULTS / "poisson_solution.xdmf"), "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_h)
    print("Solution exported to poisson_solution.xdmf (open in ParaView)")

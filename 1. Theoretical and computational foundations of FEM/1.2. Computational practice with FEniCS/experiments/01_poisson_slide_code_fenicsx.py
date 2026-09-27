import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Create mesh: unit square, 8x8 triangulation
msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

# P1 Lagrange finite elements
V = fem.functionspace(msh, ("Lagrange", 1))

# Exact solution / Dirichlet data
u_D = fem.Function(V)
u_D.interpolate(lambda x: 1 + x[0] * x[0] + 2 * x[1] * x[1])

# Identify boundary
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)

# Apply Dirichlet BC
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_D, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, default_scalar_type(-6.0))

# Bilinear form a(u,v) and linear form L(v)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solve
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="experiment1_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
u_h = problem.solve()

# Compute errors
error_L2_form = fem.form(ufl.inner(u_h - u_D, u_h - u_D) * ufl.dx)
error_L2 = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_L2_form), op=MPI.SUM))

grad_err = ufl.grad(u_h - u_D)
error_H1_semi_form = fem.form(ufl.inner(grad_err, grad_err) * ufl.dx)
error_H1_semi = np.sqrt(msh.comm.allreduce(fem.assemble_scalar(error_H1_semi_form), op=MPI.SUM))
error_H1 = np.sqrt(error_L2**2 + error_H1_semi**2)

print(f"L2 error: {error_L2:.6e}")
print(f"H1 error: {error_H1:.6e}")

# Plot
try:
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    pv.set_jupyter_backend("static")

    topology_m, cell_types_m, geometry_m = vtk_mesh(msh, tdim)
    mesh_grid = pv.UnstructuredGrid(topology_m, cell_types_m, geometry_m)

    topology_v, cell_types_v, geometry_v = vtk_mesh(V)
    solution_grid = pv.UnstructuredGrid(topology_v, cell_types_v, geometry_v)
    solution_grid.point_data["u_h"] = u_h.x.array.real

    plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 600))

    plotter.subplot(0, 0)
    plotter.add_mesh(solution_grid, scalars="u_h", cmap="viridis", show_edges=False)
    plotter.add_title("Solution u_h")
    plotter.view_xy()

    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_grid, show_edges=True, color="white", edge_color="black")
    plotter.add_title("Mesh")
    plotter.view_xy()

    plotter.screenshot(str(FIGURES / "poisson_slide_code_fenicsx.png"), scale=2)
    plotter.close()

except ImportError:
    print("Install pyvista for plotting: pip install pyvista")

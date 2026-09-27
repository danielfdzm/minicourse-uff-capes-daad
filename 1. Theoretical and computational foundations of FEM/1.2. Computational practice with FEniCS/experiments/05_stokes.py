"""
05_stokes.py — Stokes equations with Taylor-Hood elements (FEniCSx)
===================================================================

Problem (lid-driven cavity):
    -Lap(u) + grad(p) = 0   in  Omega
            div(u)     = 0   in  Omega
                 u     = (1,0) on top wall (lid)
                 u     = 0     on other walls

Uses Taylor-Hood elements: P2 for velocity, P1 for pressure
(inf-sup stable).
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
import ufl
from basix.ufl import element, mixed_element
from pathlib import Path

# ── Output folders (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# ── Mesh ──
msh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
tdim = msh.topology.dim
fdim = tdim - 1

# ── Mixed function space: Taylor-Hood P2/P1 ──
P2 = element("Lagrange", msh.topology.cell_name(), 2, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.topology.cell_name(), 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(msh, TH)

# ── Boundary conditions ──
# Extract velocity sub-space and its DOF map
V_sub, _ = W.sub(0).collapse()
Q_sub, _ = W.sub(1).collapse()

# Lid (top wall): u = (1, 0)
def lid_boundary(x):
    return np.isclose(x[1], 1.0)

lid_facets = mesh.locate_entities_boundary(msh, fdim, lid_boundary)
lid_dofs = fem.locate_dofs_topological((W.sub(0), V_sub), fdim, lid_facets)

u_lid = fem.Function(V_sub)
u_lid.interpolate(lambda x: (np.ones_like(x[0]),
                              np.zeros_like(x[0])))
bc_lid = fem.dirichletbc(u_lid, lid_dofs, W.sub(0))

# No-slip walls (bottom, left, right): u = (0, 0)
def noslip_boundary(x):
    on_left_or_right = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    on_bottom = np.isclose(x[1], 0.0)
    not_on_top = np.logical_not(np.isclose(x[1], 1.0))
    return on_bottom | (on_left_or_right & not_on_top)

noslip_facets = mesh.locate_entities_boundary(msh, fdim, noslip_boundary)
noslip_dofs = fem.locate_dofs_topological(
    (W.sub(0), V_sub), fdim, noslip_facets)

u_noslip = fem.Function(V_sub)
u_noslip.interpolate(lambda x: (np.zeros_like(x[0]),
                                 np.zeros_like(x[0])))
bc_noslip = fem.dirichletbc(u_noslip, noslip_dofs, W.sub(0))

# Pin pressure at one point to remove the constant-pressure nullspace.
def pressure_pin_point(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))

pressure_dofs = fem.locate_dofs_geometrical((W.sub(1), Q_sub), pressure_pin_point)
pressure_dofs_sub = pressure_dofs[0] if isinstance(pressure_dofs, (list, tuple)) else pressure_dofs
bc_pressure = fem.dirichletbc(default_scalar_type(0.0), pressure_dofs_sub, W.sub(1))

bcs = [bc_lid, bc_noslip, bc_pressure]

# ── Variational problem ──
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

a = (ufl.inner(ufl.grad(u), ufl.grad(v))
     - p * ufl.div(v)
     - q * ufl.div(u)) * ufl.dx

f_zero = fem.Constant(msh, (default_scalar_type(0.0),
                              default_scalar_type(0.0)))
L = ufl.inner(f_zero, v) * ufl.dx

# ── Solve ──
# Assemble and solve using PETSc
a_form = fem.form(a)
L_form = fem.form(L)

A = assemble_matrix(a_form, bcs=bcs)
A.assemble()

b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, bcs)

# Configure a direct solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)
try:
    pc.setFactorSolverType("mumps")
except PETSc.Error:
    pass

# Solve
w = fem.Function(W)
ksp.solve(b, w.x.petsc_vec)
w.x.scatter_forward()

# ── Extract sub-solutions ──
u_h = w.sub(0).collapse()
p_h = w.sub(1).collapse()
u_h.name = "velocity"
p_h.name = "pressure"

print(f"Velocity DOFs: {V_sub.dofmap.index_map.size_global}")
print(f"Pressure DOFs: {Q_sub.dofmap.index_map.size_global}")
print(f"Total DOFs:    {W.dofmap.index_map.size_global}")

# Interpolate to P1 spaces for visualization/export compatibility
V_out = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u_out = fem.Function(V_out, name="velocity")
u_out.interpolate(u_h)

Q_out = fem.functionspace(msh, ("Lagrange", 1))
p_out = fem.Function(Q_out, name="pressure")
p_out.interpolate(p_h)

# ── Visualize ──
try:
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    pv.set_jupyter_backend("static")
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1100))

    # Speed |u|
    V_speed = fem.functionspace(msh, ("Lagrange", 1))
    speed_expr = fem.Expression(
        ufl.sqrt(ufl.inner(u_out, u_out)),
        V_speed.element.interpolation_points)
    speed = fem.Function(V_speed, name="speed")
    speed.interpolate(speed_expr)
    speed_max = float(np.max(speed.x.array.real))
    speed_clim = [0.0, speed_max]

    topology_s, cell_types_s, geometry_s = vtk_mesh(V_speed)
    speed_grid = pv.UnstructuredGrid(topology_s, cell_types_s, geometry_s)
    speed_grid.point_data["speed"] = speed.x.array.real

    scalar_bar_speed = {
        "title": "|u|",
        "fmt": "%.3f",
        "title_font_size": 34,
        "label_font_size": 18,
    }
    scalar_bar_pressure = {
        "title": "p",
        "fmt": "%.3f",
        "title_font_size": 34,
        "label_font_size": 18,
    }

    # Build velocity grid once for glyphs (used in the 3D figure)
    topology_v, cell_types_v, geometry_v = vtk_mesh(V_out)
    velocity_grid = pv.UnstructuredGrid(topology_v, cell_types_v, geometry_v)
    u_vec = u_out.x.array.real.reshape((-1, msh.geometry.dim))
    u_vec3 = np.zeros((u_vec.shape[0], 3))
    u_vec3[:, :msh.geometry.dim] = u_vec
    velocity_grid.point_data["velocity"] = u_vec3
    velocity_grid.point_data["speed"] = np.linalg.norm(u_vec, axis=1)
    glyph = velocity_grid.glyph(orient="velocity", scale="speed", factor=0.09, tolerance=0.07)

    # Clean, slide-friendly 2D panel
    plotter = pv.Plotter(shape=(1, 2), window_size=(2200, 980))

    plotter.subplot(0, 0)
    plotter.add_mesh(speed_grid, scalars="speed", cmap="plasma",
                     clim=speed_clim, show_edges=False,
                     scalar_bar_args=scalar_bar_speed)
    speed_contours = speed_grid.contour(isosurfaces=12, scalars="speed")
    plotter.add_mesh(speed_contours, color="white", opacity=0.45, line_width=1.1)
    plotter.add_title("Stokes cavity: speed magnitude", font_size=40)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.12)

    # Pressure
    topology_p, cell_types_p, geometry_p = vtk_mesh(Q_out)
    pressure_grid = pv.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
    p_vals = p_out.x.array.real
    p_lo, p_hi = np.percentile(p_vals, [2.0, 98.0])
    pressure_display = np.clip(p_vals, p_lo, p_hi)
    pressure_grid.point_data["pressure"] = pressure_display

    plotter.subplot(0, 1)
    plotter.add_mesh(pressure_grid, scalars="pressure", cmap="coolwarm",
                     clim=[p_lo, p_hi],
                     show_edges=False,
                     scalar_bar_args=scalar_bar_pressure)
    plotter.add_title("Stokes cavity: pressure", font_size=40)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.12)

    plotter.screenshot(str(FIGURES / "stokes_cavity.png"), scale=3, transparent_background=True)
    plotter.close()
    print("Solution saved to stokes_cavity.png")

    # Dedicated 3D render
    warped_speed = speed_grid.warp_by_scalar("speed", factor=0.17 / max(speed_max, 1.0e-12))
    contours = speed_grid.contour(isosurfaces=12, scalars="speed")

    plotter3d = pv.Plotter(window_size=(1720, 1180))
    plotter3d.set_background("#f8fafc")
    plotter3d.enable_lightkit()
    plotter3d.enable_shadows()
    plotter3d.add_mesh(warped_speed, scalars="speed", cmap="turbo",
                       smooth_shading=True,
                       ambient=0.10,
                       diffuse=0.86,
                       specular=0.62,
                       specular_power=24,
                       scalar_bar_args=scalar_bar_speed)
    plotter3d.add_mesh(contours, color="#111827", line_width=1.4, opacity=0.42)
    plotter3d.add_mesh(glyph, color="#0f172a")
    plotter3d.add_title("Lid-driven cavity flow (3D)", font_size=42)
    # Rotate the isometric cavity view by 180 degrees around the vertical axis.
    plotter3d.camera_position = [
        (-1.85, -1.70, 1.35),
        (0.50, 0.50, 0.07),
        (0.0, 0.0, 1.0),
    ]
    plotter3d.reset_camera()
    plotter3d.camera.zoom(1.03)
    plotter3d.screenshot(str(FIGURES / "stokes_cavity_3d.png"), scale=3, transparent_background=True)
    plotter3d.close()
    print("Additional 3D plot saved to stokes_cavity_3d.png")

except ImportError:
    print("Install pyvista for visualization: pip install pyvista")

# Export to XDMF
from dolfinx.io import XDMFFile

with XDMFFile(msh.comm, str(RESULTS / "stokes_velocity.xdmf"), "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_out)

with XDMFFile(msh.comm, str(RESULTS / "stokes_pressure.xdmf"), "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(p_out)

print("Solutions exported to stokes_velocity.xdmf and stokes_pressure.xdmf")

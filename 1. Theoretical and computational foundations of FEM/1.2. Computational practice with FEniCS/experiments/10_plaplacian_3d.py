"""
10_plaplacian_3d.py
===================

Regularized p-Laplacian on a 3D unit cube.

Application model:
    pressure / hydraulic-head field for a power-law fluid moving through
    a porous 3D block.

    -div( K(x) (eps^2 + |grad u|^2)^((p-2)/2) grad u )
        + sigma*u = s(x)

with homogeneous Neumann/no-flow data on the outer boundary.  The small
sigma*u term is a leakage/reference term that removes the constant nullspace
of the pure Neumann operator.  The nonlinearity captures a gradient-dependent
mobility law.  The small eps regularizes the operator near |grad u| = 0.

The script writes:
    p_laplacian_3d.xdmf
    p_laplacian_3d.png
"""

from __future__ import annotations

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
import ufl
from pathlib import Path

# ── Output folders (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def solve_plaplacian():
    # A small 3D mesh keeps the demo quick enough for slides.
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 14, 14, 14)
    V = fem.functionspace(msh, ("Lagrange", 1))

    p_exp = 3.0
    eps = 5.0e-2
    sigma = 0.15
    x = ufl.SpatialCoordinate(msh)

    # Heterogeneous permeability and two localized wells.
    K = 0.7 + 0.6 * x[2] + 0.15 * ufl.sin(2.0 * ufl.pi * x[0])
    injection = 60.0 * ufl.exp(
        -90.0 * ((x[0] - 0.28) ** 2 + (x[1] - 0.45) ** 2 + (x[2] - 0.45) ** 2)
    )
    extraction = 42.0 * ufl.exp(
        -85.0 * ((x[0] - 0.74) ** 2 + (x[1] - 0.58) ** 2 + (x[2] - 0.50) ** 2)
    )
    source = injection - extraction

    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    grad_u = ufl.grad(u)
    mobility = (eps**2 + ufl.inner(grad_u, grad_u)) ** ((p_exp - 2.0) / 2.0)

    F = (
        K * mobility * ufl.inner(grad_u, ufl.grad(v)) * ufl.dx
        + sigma * u * v * ufl.dx
        - source * v * ufl.dx
    )
    J = ufl.derivative(F, u)

    problem = NonlinearProblem(
        F,
        u,
        bcs=[],
        J=J,
        petsc_options_prefix="plap3d_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1.0e-9,
            "snes_atol": 1.0e-10,
            "snes_max_it": 35,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    problem.solve()
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("3D p-Laplacian Newton solve did not converge")

    if msh.comm.rank == 0:
        print(f"Newton iterations: {problem.solver.getIterationNumber()}")

    with XDMFFile(msh.comm, str(RESULTS / "p_laplacian_3d.xdmf"), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(u)

    return msh, V, u


def gather_vertex_data(msh, V, u):
    coords_local = V.tabulate_dof_coordinates()[:, :3]
    vals_local = u.x.array.real[: coords_local.shape[0]]
    local = np.column_stack((coords_local, vals_local))
    gathered = msh.comm.gather(local, root=0)

    if msh.comm.rank != 0:
        return None

    data = np.vstack(gathered)
    rounded = np.round(data[:, :3], 13)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    return data[np.sort(unique)]


def gather_surface_data(msh, V, u):
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(fdim, tdim)

    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    facet_to_vertex = msh.topology.connectivity(fdim, 0)

    geometry = msh.geometry.x[:, :3]
    dof_coords = V.tabulate_dof_coordinates()[:, :3]
    dof_values = u.x.array.real[: dof_coords.shape[0]]
    value_lookup = {
        tuple(np.round(coord, 13)): value
        for coord, value in zip(dof_coords, dof_values, strict=False)
    }

    local_tri_coords = []
    local_tri_values = []
    for facet in boundary_facets:
        vertices = facet_to_vertex.links(int(facet))
        coords = geometry[vertices]
        values = np.array(
            [value_lookup[tuple(np.round(coord, 13))] for coord in coords],
            dtype=float,
        )
        local_tri_coords.append(coords)
        local_tri_values.append(values)

    local_tri_coords = np.asarray(local_tri_coords)
    local_tri_values = np.asarray(local_tri_values)
    gathered = msh.comm.gather((local_tri_coords, local_tri_values), root=0)

    if msh.comm.rank != 0:
        return None

    vertex_map = {}
    vertices = []
    values = []
    triangles = []
    for tri_coords, tri_values in gathered:
        for coords, vals in zip(tri_coords, tri_values, strict=False):
            tri = []
            for coord, value in zip(coords, vals, strict=False):
                key = tuple(np.round(coord, 13))
                if key not in vertex_map:
                    vertex_map[key] = len(vertices)
                    vertices.append(coord)
                    values.append(value)
                tri.append(vertex_map[key])
            triangles.append(tri)

    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64), np.asarray(values)


def render_surface_mesh(vertices, triangles, values, clim):
    import pyvista as pv

    faces = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    surface = pv.PolyData(vertices, faces)
    surface.point_data["u"] = values

    plotter = pv.Plotter(off_screen=True, window_size=(1000, 760))
    plotter.set_background("white")
    plotter.add_mesh(
        surface,
        scalars="u",
        cmap="viridis",
        clim=clim,
        show_edges=True,
        edge_color="#263238",
        line_width=0.55,
        interpolate_before_map=True,
        scalar_bar_args={
            "title": "u",
            "fmt": "%.2f",
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.18,
            "width": 0.08,
            "height": 0.62,
            "title_font_size": 22,
            "label_font_size": 18,
        },
    )
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.zoom(1.18)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def make_figure(data, surface_data):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    x, y, z, u = data.T
    umax = max(np.max(np.abs(u)), 1.0e-12)

    fig = plt.figure(figsize=(13.5, 4.4), constrained_layout=True)

    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    keep = np.abs(u) > 0.16 * umax
    scatter = ax0.scatter(
        x[keep],
        y[keep],
        z[keep],
        c=u[keep],
        s=18,
        cmap="viridis",
        alpha=0.78,
        linewidths=0.0,
    )
    ax0.set_title("3D pressure head")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_zlabel("z")
    ax0.view_init(elev=24, azim=-52)
    fig.colorbar(scatter, ax=ax0, shrink=0.72, pad=0.05)

    ax1 = fig.add_subplot(1, 3, 2)
    mid = np.abs(z - 0.5) <= 1.0 / 28.0
    tri = mtri.Triangulation(x[mid], y[mid])
    contour = ax1.tricontourf(tri, u[mid], levels=18, cmap="viridis")
    ax1.tricontour(tri, u[mid], levels=8, colors="white", linewidths=0.45, alpha=0.7)
    ax1.set_aspect("equal")
    ax1.set_title(r"Mid-plane slice $z \approx 0.5$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(contour, ax=ax1, shrink=0.82)

    ax2 = fig.add_subplot(1, 3, 3)
    surface_vertices, surface_triangles, surface_values = surface_data
    surface_min = float(np.min(surface_values))
    surface_max = float(np.max(surface_values))
    if np.isclose(surface_min, surface_max):
        padding = max(abs(surface_min), 1.0) * 1.0e-12
        surface_min -= padding
        surface_max += padding
    surface_image = render_surface_mesh(
        surface_vertices,
        surface_triangles,
        surface_values,
        clim=(surface_min, surface_max),
    )
    ax2.imshow(surface_image)
    ax2.set_title("Boundary mesh colored by solution")
    ax2.axis("off")

    fig.suptitle("Regularized p-Laplacian in a 3D porous block", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES / "p_laplacian_3d.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    msh, V, u = solve_plaplacian()
    data = gather_vertex_data(msh, V, u)
    surface_data = gather_surface_data(msh, V, u)
    if msh.comm.rank == 0:
        make_figure(data, surface_data)
        print("Saved p_laplacian_3d.png and p_laplacian_3d.xdmf")


if __name__ == "__main__":
    main()

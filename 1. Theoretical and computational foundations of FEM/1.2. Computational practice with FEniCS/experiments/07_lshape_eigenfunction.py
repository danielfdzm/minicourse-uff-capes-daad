"""
07_lshape_eigenfunction.py — Laplacian eigenfunction on an L-shaped domain
=========================================================================

Computes Dirichlet eigenpairs for

    -Delta(phi) = lambda * phi    in Omega
                 phi = 0          on dOmega

on an L-shaped domain in 2D and visualizes the first eigenfunction:
1) in 2D (top view with contours)
2) in 3D (warped surface, MATLAB-logo-style view)
"""

import numpy as np
from mpi4py import MPI
import gmsh

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmsh as io_gmsh
import ufl

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)


def create_lshape_mesh(comm: MPI.Comm, h: float = 0.045, L: float = 1.0):
    """Create an L-shaped mesh: [-L,L]^2 without [0,L] x [-L,0]."""
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("lshape")

        # Explicitly build a square outer box.
        outer = gmsh.model.occ.addRectangle(-L, -L, 0.0, 2.0 * L, 2.0 * L)
        cutout = gmsh.model.occ.addRectangle(0.0, -L, 0.0, L, L)
        domain, _ = gmsh.model.occ.cut([(2, outer)], [(2, cutout)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        surface_tag = domain[0][1]
        gmsh.model.addPhysicalGroup(2, [surface_tag], 1)
        gmsh.model.setPhysicalName(2, 1, "Omega")

        boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
        boundary_tags = [tag for dim, tag in boundary if dim == 1]
        gmsh.model.addPhysicalGroup(1, boundary_tags, 2)
        gmsh.model.setPhysicalName(1, 2, "Boundary")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

    mesh_data = io_gmsh.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)

    if comm.rank == 0:
        gmsh.finalize()

    return mesh_data.mesh


def petsc_to_scipy(A):
    """Convert a PETSc matrix to SciPy CSR."""
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.getSize())


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run this script with a single MPI rank (mpiexec -n 1).")

    L = 1.0
    # Keep computation on [-L, L]^2, but cut only the 3D view to [-6L/7, 6L/7]^2.
    plot_cut_half = (6.0 / 7.0) * L

    # ── 1) Build L-shaped mesh and FE space ──
    msh = create_lshape_mesh(comm, h=0.04, L=L)
    V = fem.functionspace(msh, ("Lagrange", 2))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    m = u * v * ufl.dx

    # ── 2) Assemble stiffness/mass matrices ──
    A_petsc = assemble_matrix(fem.form(a))
    A_petsc.assemble()

    M_petsc = assemble_matrix(fem.form(m))
    M_petsc.assemble()

    # Dirichlet condition handled by reduction to interior dofs
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    num_dofs = V.dofmap.index_map.size_global
    free_dofs = np.setdiff1d(np.arange(num_dofs, dtype=np.int32), boundary_dofs)

    A = petsc_to_scipy(A_petsc)
    M = petsc_to_scipy(M_petsc)
    A_free = A[free_dofs][:, free_dofs].tocsr()
    M_free = M[free_dofs][:, free_dofs].tocsr()

    print(f"Global dofs:   {num_dofs}")
    print(f"Boundary dofs: {len(boundary_dofs)}")
    print(f"Interior dofs: {len(free_dofs)}")

    # ── 3) Solve generalized eigenproblem ──
    k = min(8, A_free.shape[0] - 2)
    if k < 1:
        raise RuntimeError("Not enough interior dofs for eigenvalue solve.")

    eigvals, eigvecs_free = eigsh(A_free, k=k, M=M_free, sigma=0.0, which="LM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs_free = eigvecs_free[:, order]

    print("\nLowest eigenvalues on the L-shaped domain:")
    for i, lam in enumerate(eigvals[:5], start=1):
        print(f"  lambda_{i} = {lam:.8f}")

    # First eigenfunction
    phi = np.zeros(num_dofs, dtype=np.float64)
    phi[free_dofs] = eigvecs_free[:, 0]
    max_abs = np.max(np.abs(phi))
    if max_abs > 0:
        phi /= max_abs
    if phi[np.argmax(np.abs(phi))] < 0:
        phi *= -1.0

    phi_h = fem.Function(V, name="phi_1")
    phi_h.x.array[:] = phi
    phi_h.x.scatter_forward()

    # ── 4) Visualize in 2D and 3D ──
    try:
        import pyvista as pv
        from dolfinx.plot import vtk_mesh
        from matplotlib.colors import LinearSegmentedColormap

        pv.set_jupyter_backend("static")

        topology, cell_types, geometry = vtk_mesh(V)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["phi"] = phi_h.x.array.real

        # 3D cut only: keep geometry unchanged in 2D, cut 3D view to [-6L/7, 6L/7]^2.
        pts = grid.points
        point_ids = np.where(
            (np.abs(pts[:, 0]) <= plot_cut_half)
            & (np.abs(pts[:, 1]) <= plot_cut_half)
        )[0]
        grid_3d = grid.extract_points(point_ids, adjacent_cells=True, include_cells=True)

        amp = float(np.max(np.abs(grid.point_data["phi"])))
        clim = [0.0, amp]
        matlab_logo_cmap = LinearSegmentedColormap.from_list(
            "matlab_logo",
            [
                "#1a2b8f",  # deep blue
                "#1f77d0",  # azure
                "#31b3df",  # cyan
                "#f2d35e",  # warm yellow
                "#ea8b2d",  # orange
                "#be4a1d",  # deep orange-red
            ],
        )

        # Missing quarter [0,1] x [-1,0], shown as z=0 with phi=0 color.
        patch_points = np.array(
            [
                [0.0, -L, 0.0],
                [L, -L, 0.0],
                [L, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        patch_faces = np.hstack([[3, 0, 1, 2], [3, 0, 2, 3]])
        missing_patch_2d = pv.PolyData(patch_points, patch_faces)
        missing_patch_2d.point_data["phi"] = np.zeros(4, dtype=np.float64)

        patch_points_3d = np.array(
            [
                [0.0, -plot_cut_half, 0.0],
                [plot_cut_half, -plot_cut_half, 0.0],
                [plot_cut_half, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        missing_patch_3d = pv.PolyData(patch_points_3d, patch_faces)
        missing_patch_3d.point_data["phi"] = np.zeros(4, dtype=np.float64)

        # 2D plot
        plotter2d = pv.Plotter(window_size=(1460, 900))
        plotter2d.set_background("#f8fafc")
        scalar_bar_args = {
            "title": "phi_1",
            "fmt": "%.2f",
            "position_x": 0.89,
            "position_y": 0.14,
            "width": 0.055,
            "height": 0.72,
            "title_font_size": 20,
            "label_font_size": 14,
        }
        plotter2d.add_mesh(
            missing_patch_2d,
            scalars="phi",
            cmap=matlab_logo_cmap,
            clim=clim,
            show_edges=False,
            lighting=False,
            show_scalar_bar=False,
        )
        plotter2d.add_mesh(
            grid,
            scalars="phi",
            cmap=matlab_logo_cmap,
            clim=clim,
            show_edges=False,
            scalar_bar_args=scalar_bar_args,
        )
        contour = grid.contour(isosurfaces=18, scalars="phi")
        plotter2d.add_mesh(contour, color="white", line_width=1.0, opacity=0.45)
        plotter2d.add_title("L-shaped domain: first eigenfunction (2D)", font_size=40)
        plotter2d.view_xy()
        plotter2d.enable_parallel_projection()
        plotter2d.reset_camera()
        plotter2d.camera.zoom(1.09)
        plotter2d.screenshot(str(FIGURES / "lshape_eigenfunction_2d.png"), scale=3, transparent_background=True)
        plotter2d.close()

        # 3D plot (MATLAB-logo-style)
        # Double vertical exaggeration so the highest point is 2x the previous height.
        warp_factor = 1.50 / max(amp, 1.0e-12)
        warped = grid_3d.warp_by_scalar("phi", factor=warp_factor)

        plotter3d = pv.Plotter(window_size=(1450, 980))
        plotter3d.set_background("#f5f6f8")
        plotter3d.enable_lightkit()
        plotter3d.add_mesh(
            missing_patch_3d,
            color="#4f7fd6",
            smooth_shading=True,
            ambient=0.24,
            diffuse=0.68,
            specular=0.28,
            specular_power=14,
            show_edges=False,
            show_scalar_bar=False,
        )
        plotter3d.add_mesh(
            warped,
            color="#d74a1f",
            smooth_shading=True,
            ambient=0.22,
            diffuse=0.70,
            specular=0.30,
            specular_power=16,
            show_scalar_bar=False,
        )
        plotter3d.add_title("L-shaped Laplacian eigenfunction (MATLAB-style 3D)", font_size=42)
        plotter3d.camera_position = [(1.45, 1.55, 1.1), (-0.025, 0.0, 0.04), (0.0, 0.0, 1.0)]
        plotter3d.reset_camera()
        plotter3d.camera.zoom(1.02)
        plotter3d.screenshot(str(FIGURES / "lshape_eigenfunction_3d_matlab.png"), scale=3, transparent_background=True)
        plotter3d.close()

        print("\nSaved lshape_eigenfunction_2d.png")
        print("Saved lshape_eigenfunction_3d_matlab.png")

    except ImportError:
        print("Install pyvista for plotting: pip install pyvista")

    # ── 5) Clean up PETSc matrices ──
    A_petsc.destroy()
    M_petsc.destroy()
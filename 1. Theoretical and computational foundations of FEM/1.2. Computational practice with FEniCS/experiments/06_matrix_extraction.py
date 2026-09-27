"""
06_matrix_extraction.py — Extracting matrices for SciPy (FEniCSx)
=================================================================

Shows how to assemble FEniCSx bilinear forms into PETSc
matrices, convert to SciPy sparse format, and solve an
eigenvalue problem for the Laplacian on the unit square.
"""

import numpy as np
import itertools
from mpi4py import MPI

from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix
import ufl

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# ── Setup ──
msh = mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
if msh.comm.size != 1:
    raise RuntimeError(
        "This script converts PETSc matrices to SciPy and must be run with a single MPI rank "
        "(e.g. `mpiexec -n 1 python 06_matrix_extraction.py`).")

V = fem.functionspace(msh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Stiffness and mass forms
a_stiff = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
a_mass = u * v * ufl.dx

# Identify constrained (boundary) and free (interior) dofs.
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(msh.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
num_dofs = V.dofmap.index_map.size_global
free_dofs = np.setdiff1d(np.arange(num_dofs, dtype=np.int32), boundary_dofs)

# ── Assemble into PETSc matrices ──
A_petsc = assemble_matrix(fem.form(a_stiff))
A_petsc.assemble()

M_petsc = assemble_matrix(fem.form(a_mass))
M_petsc.assemble()

# ── Convert PETSc matrix to SciPy CSR ──
def petsc_to_scipy(A):
    """Convert a PETSc Mat to SciPy CSR sparse matrix."""
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.getSize())

A_sp = petsc_to_scipy(A_petsc)
M_sp = petsc_to_scipy(M_petsc)
A_free = A_sp[free_dofs][:, free_dofs].tocsr()
M_free = M_sp[free_dofs][:, free_dofs].tocsr()

print(f"Stiffness matrix: {A_sp.shape}, nnz = {A_sp.nnz}")
print(f"Mass matrix:      {M_sp.shape}, nnz = {M_sp.nnz}")
print(f"Sparsity:         {A_sp.nnz / np.prod(A_sp.shape) * 100:.2f}%")
print(f"Reduced matrix:   {A_free.shape}, nnz = {A_free.nnz}")

# ── Solve generalized eigenvalue problem ──
# Find smallest eigenvalues of:  A u = lambda M u
# (Laplacian eigenvalues on the unit square)
k = min(10, A_free.shape[0] - 2)  # number of eigenvalues
if k < 1:
    raise RuntimeError("Not enough interior degrees of freedom for eigenvalue solve.")

eigenvalues, eigenvectors_free = eigsh(A_free, k=k, M=M_free,
                                       sigma=0.0, which='LM')

# ── Compare with exact eigenvalues ──
# Exact eigenvalues: pi^2 * (m^2 + n^2) for m,n = 1,2,...
exact = sorted([np.pi**2 * (m**2 + n**2)
                for m, n in itertools.product(range(1, 6), repeat=2)])[:k]

print(f"\nFirst {k} Laplacian eigenvalues on [0,1]^2:")
print(f"  {'Computed':>12s}   {'Exact':>12s}   {'Rel Error':>12s}")
print(f"  {'-'*42}")

sorted_indices = np.argsort(eigenvalues)
sorted_evals = eigenvalues[sorted_indices]
for comp, ex in zip(sorted_evals, exact):
    rel_err = abs(comp - ex) / ex
    print(f"  {comp:12.6f}   {ex:12.6f}   {rel_err:.2e}")

# ── Visualize eigenfunctions ──
try:
    import pyvista as pv
    from dolfinx.plot import vtk_mesh

    # Plot the first 4 eigenfunctions
    n_plot = min(4, k)

    pv.set_jupyter_backend("static")
    plotter = pv.Plotter(shape=(2, n_plot),
                         window_size=(420 * n_plot, 860))

    for i in range(n_plot):
        idx = sorted_indices[i]
        lam = eigenvalues[idx]

        # Set eigenvector values on the FE function
        u_eig = fem.Function(V, name=f"eig_{i+1}")
        full_eig = np.zeros(num_dofs, dtype=eigenvectors_free.dtype)
        full_eig[free_dofs] = eigenvectors_free[:, idx]
        u_eig.x.array[:] = full_eig.real
        amp = float(np.max(np.abs(u_eig.x.array.real)))
        clim = [-amp, amp]
        warp_scale = 0.16 / max(amp, 1.0e-12)

        topology, cell_types, geometry = vtk_mesh(V)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data["eig"] = u_eig.x.array.real

        plotter.subplot(0, i)
        plotter.add_mesh(grid, scalars="eig", cmap="RdBu_r",
                         clim=clim, show_edges=False,
                         scalar_bar_args={"title": "phi"} if i == n_plot - 1 else None)
        plotter.add_title(f"2D mode {i+1}, lambda={lam:.2f}", font_size=22)
        plotter.view_xy()

        plotter.subplot(1, i)
        warped = grid.warp_by_scalar("eig", factor=warp_scale)
        contours = grid.contour(isosurfaces=10, scalars="eig")
        plotter.add_mesh(warped, scalars="eig", cmap="RdBu_r",
                         clim=clim, smooth_shading=True, specular=0.25,
                         scalar_bar_args={"title": "phi"} if i == n_plot - 1 else None)
        plotter.add_mesh(contours, color="white", opacity=0.55, line_width=1.0)
        plotter.add_title(f"3D mode {i+1}", font_size=22)
        plotter.view_isometric()
        plotter.reset_camera()

    plotter.screenshot(str(FIGURES / "eigenfunctions.png"), scale=3, transparent_background=True)
    plotter.close()
    print(f"\nEigenfunctions saved to eigenfunctions.png")

    # Dedicated highlight of first mode in 3D
    idx0 = sorted_indices[0]
    u0 = fem.Function(V, name="eig_1")
    full0 = np.zeros(num_dofs, dtype=eigenvectors_free.dtype)
    full0[free_dofs] = eigenvectors_free[:, idx0]
    u0.x.array[:] = full0.real
    amp0 = float(np.max(np.abs(u0.x.array.real)))
    grid0 = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid0.point_data["eig"] = u0.x.array.real
    warped0 = grid0.warp_by_scalar("eig", factor=0.2 / max(amp0, 1.0e-12))

    plotter3d = pv.Plotter(window_size=(1250, 900))
    plotter3d.set_background("#f8fafc")
    plotter3d.add_mesh(warped0, scalars="eig", cmap="coolwarm",
                       smooth_shading=True, specular=0.35,
                       scalar_bar_args={"title": "phi_1", "fmt": "%.3f"})
    plotter3d.add_title("First eigenmode (3D)", font_size=36)
    plotter3d.view_isometric()
    plotter3d.reset_camera()
    plotter3d.camera.zoom(0.92)
    plotter3d.screenshot(str(FIGURES / "eigenmode1_3d.png"), scale=3, transparent_background=True)
    plotter3d.close()
    print("Saved eigenmode1_3d.png")

except ImportError:
    print("\nInstall pyvista for visualization: pip install pyvista")

# ── Also plot eigenvalue comparison as a bar chart ──
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(k)
width = 0.35
rel_errors = np.abs(sorted_evals - np.array(exact)) / np.array(exact)

ax.bar(x_pos - width/2, sorted_evals, width,
    label='Computed', color='#2563eb')
ax.bar(x_pos + width/2, exact, width,
    label='Exact', color='#f97316', alpha=0.78)

ax_rel = ax.twinx()
ax_rel.plot(x_pos, rel_errors, 'o--', color='#111827', linewidth=2,
         markersize=5, label='Relative error')

ax.set_xlabel('Eigenvalue index', fontsize=12)
ax.set_ylabel('$\\lambda$', fontsize=14)
ax.set_title('Laplacian Eigenvalues on $[0,1]^2$', fontsize=30)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'$\\lambda_{{{i+1}}}$' for i in range(k)])
ax_rel.set_ylabel('Relative error', fontsize=12)
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_rel.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "eigenvalue_comparison.png", dpi=260, bbox_inches='tight', transparent=True)
plt.close(fig)

# Compare computed and exact eigenvalues
fig_parity, ax_parity = plt.subplots(figsize=(6.8, 6.2))
ax_parity.scatter(exact, sorted_evals, s=78, color="#0ea5e9", edgecolors="black", alpha=0.86)
diag_min = min(min(exact), float(np.min(sorted_evals)))
diag_max = max(max(exact), float(np.max(sorted_evals)))
ax_parity.plot([diag_min, diag_max], [diag_min, diag_max], '--', color='#ef4444', linewidth=2)
ax_parity.set_xlabel('Exact eigenvalue')
ax_parity.set_ylabel('Computed eigenvalue')
ax_parity.set_title('Computed vs exact eigenvalue parity', fontsize=28, fontweight='bold')
ax_parity.grid(True, alpha=0.3)
fig_parity.savefig(FIGURES / "eigenvalue_parity.png", dpi=260, bbox_inches='tight', transparent=True)
plt.close(fig_parity)

print("Saved eigenvalue_comparison.png and eigenvalue_parity.png")

# ── Clean up PETSc objects ──
A_petsc.destroy()
M_petsc.destroy()

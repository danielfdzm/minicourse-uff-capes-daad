"""
08_neumann_refinement.py — Local mesh refinement for an elliptic Neumann problem
===============================================================================

Elliptic model problem on an L-shaped domain:

    -Delta u + u = f      in Omega
          du/dn  = 0      on dOmega   (homogeneous Neumann)

with a singular right-hand side concentrated at the re-entrant corner:

    f(x, y) = 1 / (x^2 + y^2 + eps^2)^(3/4)

Omega = [-1, 1]^2 \\ [0, 1] x [-1, 0]

The re-entrant corner + singular forcing produce steep gradients near (0, 0),
so local mesh refinement in that region is beneficial.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh
import ufl

from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as io_gmsh
from dolfinx.plot import vtk_mesh
from pathlib import Path

# ── Output folder (next to this script, whatever the working directory) ──
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)


def create_lshape_mesh_with_corner_refinement(
    comm: MPI.Comm,
    h_far: float,
    h_near: float,
    r_min: float = 0.05,
    r_max: float = 0.35,
):
    """Create an L-shaped mesh with a size field refining near (0, 0)."""
    if comm.rank == 0:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("lshape_local_refinement")

        # L-shape: square minus lower-right quadrant
        outer = gmsh.model.occ.addRectangle(-1.0, -1.0, 0.0, 2.0, 2.0)
        cutout = gmsh.model.occ.addRectangle(0.0, -1.0, 0.0, 1.0, 1.0)
        domain, _ = gmsh.model.occ.cut([(2, outer)], [(2, cutout)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # Auxiliary point at the singular/re-entrant corner.
        p0 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        gmsh.model.occ.synchronize()

        # Optional physical tag for the domain surface.
        surface_tag = domain[0][1]
        gmsh.model.addPhysicalGroup(2, [surface_tag], 1)
        gmsh.model.setPhysicalName(2, 1, "Omega")

        # Distance-based local size field around corner (0,0).
        fdist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fdist, "PointsList", [p0])

        fth = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fth, "InField", fdist)
        gmsh.model.mesh.field.setNumber(fth, "SizeMin", h_near)
        gmsh.model.mesh.field.setNumber(fth, "SizeMax", h_far)
        gmsh.model.mesh.field.setNumber(fth, "DistMin", r_min)
        gmsh.model.mesh.field.setNumber(fth, "DistMax", r_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(fth)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min(h_near, h_far))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max(h_near, h_far))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

    mesh_data = io_gmsh.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)

    if comm.rank == 0:
        gmsh.finalize()

    return mesh_data.mesh


def solve_neumann_screened_poisson(msh, level_idx: int):
    """Solve -Delta u + u = f with homogeneous Neumann BC on msh."""
    V = fem.functionspace(msh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    # Stronger singular forcing near the re-entrant corner to better expose
    # the benefit of local refinement in both solution and error-metric plots.
    eps = 0.006
    r2 = x[0] * x[0] + x[1] * x[1] + eps * eps
    f_expr = 1.0 / (r2 ** 0.75)

    # Neumann BC is natural in the weak form; no Dirichlet constraints here.
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * ufl.dx
    L = f_expr * v * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options_prefix=f"neumann_ref_{level_idx}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    u_h = problem.solve()
    u_h.name = "u_h"

    # Corner-weighted error metric emphasizing the singular neighborhood.
    weight = ufl.exp(-(x[0] * x[0] + x[1] * x[1]) / (0.045**2))
    metric_form = fem.form(weight * u_h * ufl.dx)
    error_metric = msh.comm.allreduce(fem.assemble_scalar(metric_form), op=MPI.SUM)

    ndofs = V.dofmap.index_map.size_global

    return V, u_h, ndofs, error_metric


def extract_triangulation_for_mesh(msh):
    tdim = msh.topology.dim
    topology, _, geometry = vtk_mesh(msh, tdim)
    cells = topology.reshape((-1, 4))[:, 1:4]
    tri = mtri.Triangulation(geometry[:, 0], geometry[:, 1], cells)
    return tri


def extract_triangulation_for_solution(V, values):
    topology, _, geometry = vtk_mesh(V)
    cells = topology.reshape((-1, 4))[:, 1:4]
    tri = mtri.Triangulation(geometry[:, 0], geometry[:, 1], cells)
    return tri, values


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Run this script with a single MPI rank (mpiexec -n 1).")

    # Keep far-field mesh size mostly fixed while refining near singular corner.
    h_far = 0.22
    # Start from a mesh twice as coarse as the previous first level (0.11 -> 0.22).
    local_h_near = [0.22, 0.11, 0.06, 0.032, 0.018]

    levels = []

    for i, h_near in enumerate(local_h_near):
        msh = create_lshape_mesh_with_corner_refinement(comm, h_far=h_far, h_near=h_near)
        V, u_h, ndofs, error_metric = solve_neumann_screened_poisson(msh, level_idx=i)

        tri_mesh = extract_triangulation_for_mesh(msh)
        tri_sol, sol_vals = extract_triangulation_for_solution(V, u_h.x.array.real.copy())

        levels.append(
            {
                "idx": i,
                "h_near": h_near,
                "ndofs": ndofs,
                "error_metric": error_metric,
                "tri_mesh": tri_mesh,
                "tri_sol": tri_sol,
                "u_vals": sol_vals,
            }
        )

        print(f"Level {i}: h_near={h_near:.4f}, dofs={ndofs}, error_metric={error_metric:.8e}")

    # Compare error metric against the finest local refinement level.
    metric_ref = levels[-1]["error_metric"]
    print("\nError metric differences vs finest level:")
    for data in levels:
        err = abs(data["error_metric"] - metric_ref)
        print(f"  Level {data['idx']}: |E - E_ref| = {err:.6e}")

    # --- Plot 1: mesh refinement + solution snapshots ---
    nlev = len(levels)
    fig, axes = plt.subplots(2, nlev, figsize=(5.2 * nlev, 9.0), constrained_layout=True)

    if nlev == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    corner_window = 0.30
    all_u = np.concatenate([d["u_vals"] for d in levels])
    u_min = float(np.percentile(all_u, 2.0))
    u_max = float(np.percentile(all_u, 99.5))
    if u_max <= u_min:
        u_min = float(np.min(all_u))
        u_max = float(np.max(all_u))
    levels_cf = np.linspace(u_min, u_max, 36)
    cf_last = None

    for j, d in enumerate(levels):
        # Mesh panel
        axm = axes[0, j]
        axm.triplot(d["tri_mesh"], color="#374151", linewidth=0.42)
        axm.set_aspect("equal", adjustable="box")
        axm.set_title(
            f"Mesh L{d['idx']}\n$h_{{near}}$={d['h_near']:.3f}, dofs={d['ndofs']}",
            fontsize=24,
            fontweight="bold",
        )
        axm.set_xlim(-corner_window, corner_window)
        axm.set_ylim(-corner_window, corner_window)
        axm.scatter([0.0], [0.0], s=16, color="#ef4444", zorder=5)
        axm.set_xlabel("x", fontsize=12)
        if j == 0:
            axm.set_ylabel("y", fontsize=12)
        axm.tick_params(labelsize=10)

        # Solution panel
        axs = axes[1, j]
        u_plot = np.clip(d["u_vals"], u_min, u_max)
        cf_last = axs.tricontourf(d["tri_sol"], u_plot, levels=levels_cf, cmap="inferno", extend="both")
        axs.tricontour(d["tri_sol"], u_plot, levels=12, colors="white", linewidths=0.75, alpha=0.55)
        axs.set_aspect("equal", adjustable="box")
        axs.set_title(f"Solution $u_h$ (L{d['idx']})", fontsize=24, fontweight="bold")
        axs.set_xlim(-corner_window, corner_window)
        axs.set_ylim(-corner_window, corner_window)
        axs.scatter([0.0], [0.0], s=16, color="#facc15", zorder=5)
        axs.set_xlabel("x", fontsize=12)
        if j == 0:
            axs.set_ylabel("y", fontsize=12)
        axs.tick_params(labelsize=10)

    cbar = fig.colorbar(cf_last, ax=axes[1, :], location="right", shrink=0.88, pad=0.02)
    cbar.set_label("$u_h$", fontsize=14)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "Local mesh refinement near singular corner (zoomed view)",
        fontsize=34,
        fontweight="bold",
    )
    fig.savefig(FIGURES / "neumann_refinement.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    # --- Plot 2: corner-focused error metric vs DOFs ---
    dofs = np.array([d["ndofs"] for d in levels], dtype=float)
    q_err = np.array([abs(d["error_metric"] - metric_ref) for d in levels], dtype=float)

    fig2, ax2 = plt.subplots(figsize=(8.4, 6.0), constrained_layout=True)
    ax2.loglog(dofs[:-1], q_err[:-1], "o-", color="#1d4ed8", linewidth=2.6,
               markersize=8, markeredgecolor="black", markeredgewidth=0.7)
    ax2.set_xlabel("Global DOFs", fontsize=14)
    ax2.set_ylabel(r"$|E - E_{ref}|$", fontsize=14)
    ax2.set_title("Corner-focused error metric vs refinement", fontsize=30, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)

    fig2.savefig(FIGURES / "neumann_refinement_error_metric.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig2)

    print("\nSaved neumann_refinement.png")
    print("Saved neumann_refinement_error_metric.png")

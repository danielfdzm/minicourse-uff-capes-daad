"""Chapter 1 animation: P1 finite elements for the L-shaped eigenproblem under refinement."""
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from style import (PAPER, BLUE, INK, SECONDARY, MUTED, LABEL,
                   SOLUTION_CMAP, use_course_fonts, fig_to_image, save_gif, style_axes)
from fem_lshape import first_eigenpair, LAMBDA_1

LEVELS = (3, 4, 6, 8, 12, 16, 24, 32)
SIZE = (800, 450)
Z_SCALE = 0.95
LIGHT = np.array([-0.45, -0.55, 0.85]) / np.linalg.norm([-0.45, -0.55, 0.85])
L_OUTLINE = np.array([(-1, -1), (0, -1), (0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1)], float)


def surface_polys(pts, tris, u):
    verts = np.column_stack([pts, Z_SCALE * u])[tris]
    normals = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[normals[:, 2] < 0] *= -1
    shade = 0.85 + 0.15 * np.clip(normals @ LIGHT, 0.0, 1.0)
    colours = SOLUTION_CMAP(u[tris].mean(axis=1))[:, :3] * shade[:, None]
    return verts, np.clip(colours, 0, 1)


def draw_frame(sol, k, azim, history):
    pts, tris, u, lam, dofs, n = sol
    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor=PAPER)

    # 3D surface
    ax = fig.add_axes([-0.04, -0.06, 0.72, 0.98], projection="3d", facecolor=PAPER)
    ax.set_axis_off()
    verts, colours = surface_polys(pts, tris, u)
    # mesh lines while they are informative; the finest meshes read as a smooth surface
    edge_alpha = 0.55 if n <= 8 else 0.30 if n <= 16 else 0.0
    ax.add_collection3d(Poly3DCollection(verts, facecolors=colours, edgecolors=(1, 1, 1, edge_alpha),
                                         linewidths=0.35 if edge_alpha else 0.0))
    floor = np.column_stack([L_OUTLINE, np.zeros(len(L_OUTLINE))])
    ax.add_collection3d(Line3DCollection([floor], colors=[LABEL], linewidths=1.0, alpha=0.6))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 0.5), zoom=1.0)
    ax.view_init(elev=30, azim=azim)

    # headline
    fig.text(0.045, 0.905, "CHAPTER 1  ·  FINITE ELEMENTS", color=LABEL, fontsize=10,
             fontweight="bold")
    fig.text(0.045, 0.835, "Mesh refinement and eigenvalue convergence", color=INK,
             fontsize=16, fontweight="normal")
    fig.text(0.045, 0.785, r"First Dirichlet eigenfunction of $-\Delta u=\lambda u$ on an L-shaped domain, P1 elements",
             color=SECONDARY, fontsize=10)
    fig.text(0.70, 0.125, f"h = 1/{n}   ·   {dofs:,} unknowns", color=INK, fontsize=11)
    fig.text(0.70, 0.07, rf"$\lambda_{{1,h}}$ = {lam:.4f}", color=SECONDARY, fontsize=11)

    # convergence chart: one series, so no legend box - the title names it
    cx = fig.add_axes([0.70, 0.25, 0.265, 0.43])
    style_axes(cx)
    cx.set_xscale("log"); cx.set_yscale("log")
    cx.set_xlim(12, 5000); cx.set_ylim(1.5e-2, 3.0)
    done = history[:k + 1]
    xs, ys = [d for d, _ in done], [e for _, e in done]
    cx.plot(xs, ys, color=BLUE, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    cx.scatter(xs[:-1], ys[:-1], s=34, color=BLUE, edgecolors=PAPER, linewidths=1.6, zorder=3)
    cx.scatter(xs[-1:], ys[-1:], s=48, color=BLUE, edgecolors=INK, linewidths=1.8, zorder=4)
    cx.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(8, 6), textcoords="offset points",
                color=INK, fontsize=9)
    cx.set_xlabel("unknowns", color=MUTED, fontsize=9, labelpad=3)
    fig.text(0.70, 0.735, r"Eigenvalue error $\lambda_{1,h}-\lambda_1$", color=INK, fontsize=10.5,
             fontweight="bold")
    fig.text(0.70, 0.695, r"$\lambda_1 \approx 9.6397$", color=MUTED, fontsize=9)
    img = fig_to_image(fig, SIZE)
    plt.close(fig)
    return img


def main(out):
    use_course_fonts()
    sols = []
    for n in LEVELS:
        pts, tris, u, lam, dofs = first_eigenpair(n)
        sols.append((pts, tris, u, lam, dofs, n))
    history = [(s[4], s[3] - LAMBDA_1) for s in sols]
    frames, durations = [], []
    # Keep the camera fixed so only the numerical refinement changes.
    for k, sol in enumerate(sols):
        frames.append(draw_frame(sol, k, -105, history))
        durations.append(1100)
    durations[0] = 1800
    durations[-1] = 4000
    save_gif(frames, out, durations)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fem_eigenmode.gif")

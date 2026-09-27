"""A simple course masthead with a computed FEM surface."""
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from style import PAPER, INK, SECONDARY, GRID, SOLUTION_CMAP, use_course_fonts, fig_to_image
from fem_lshape import first_eigenpair

W_IN, H_IN, DPI = 12.0, 3.0, 200


def fem_surface(fig):
    pts, tris, u, _, _ = first_eigenpair(10)
    verts = np.column_stack([pts, 0.95 * u])[tris]
    colours = SOLUTION_CMAP(u[tris].mean(1))
    ax = fig.add_axes([0.70, 0.05, 0.28, 0.90], projection="3d", facecolor=PAPER)
    ax.set_axis_off()
    ax.add_collection3d(Poly3DCollection(verts, facecolors=colours,
                                         edgecolors=(1, 1, 1, 0.55), linewidths=0.35))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=28, azim=-112)


def main(out):
    use_course_fonts()
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=PAPER)
    fem_surface(fig)
    fig.text(0.035, 0.82, "MINICOURSE  ·  UFF  ·  CAPES/DAAD", color=SECONDARY, fontsize=11)
    fig.text(0.035, 0.59, "From Finite Elements", color=INK, fontsize=30)
    fig.text(0.035, 0.38, "to Neural PDE Solvers", color=INK, fontsize=30)
    fig.text(0.035, 0.17, "FEM theory  ·  FEniCS practice  ·  machine learning  ·  neural PDEs",
             color=SECONDARY, fontsize=11)
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.04, 0.04], color=GRID,
                              linewidth=0.8, transform=fig.transFigure))
    img = fig_to_image(fig)
    plt.close(fig)
    img.save(out, optimize=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "banner.png")

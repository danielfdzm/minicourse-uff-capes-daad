"""README banner in the style of the slide title pages: navy, mesh motif, a computed FEM surface."""
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from style import NAVY, INK_ON_DARK, INK2_ON_DARK, KICKER, TEAL_ON_DARK, use_course_fonts, fig_to_image
from fem_lshape import first_eigenpair

W_IN, H_IN, DPI = 12.0, 3.6, 200
LIGHT = np.array([-0.45, -0.55, 0.85]) / np.linalg.norm([-0.45, -0.55, 0.85])


def mesh_motif(fig):
    """Squares with one diagonal, white at 10 % opacity, as on the title slides."""
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_xlim(0, W_IN); ax.set_ylim(0, H_IN); ax.set_axis_off()
    side = 0.62
    for i in range(12):
        x0 = W_IN - (i + 1) * side
        fade = max(0.0, 1.0 - i / 11.0)            # fade out towards the title
        for j in range(7):
            y0 = j * side - 0.12
            ax.add_patch(plt.Rectangle((x0, y0), side, side, fill=False, lw=0.8,
                                       ec=(1, 1, 1, 0.10 * fade)))
            ax.plot([x0, x0 + side], [y0, y0 + side], color=(1, 1, 1, 0.10 * fade), lw=0.8)
    return ax


def fem_surface(fig):
    pts, tris, u, lam, dofs = first_eigenpair(10)
    verts = np.column_stack([pts, 0.95 * u])[tris]
    n = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n[n[:, 2] < 0] *= -1
    shade = 0.58 + 0.42 * np.clip(n @ LIGHT, 0, 1)
    colours = np.clip(TEAL_ON_DARK(u[tris].mean(1))[:, :3] * shade[:, None], 0, 1)
    ax = fig.add_axes([0.565, -0.34, 0.47, 1.62], projection="3d", facecolor="none")
    ax.set_axis_off()
    ax.add_collection3d(Poly3DCollection(verts, facecolors=colours, edgecolors=(1, 1, 1, 0.42), linewidths=0.45))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=28, azim=-112)


def main(out):
    use_course_fonts()
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=NAVY)
    mesh_motif(fig)
    fem_surface(fig)
    fig.text(0.045, 0.76, "MINICOURSE   ·   UFF   ·   CAPES/DAAD", color=KICKER, fontsize=12.5,
             fontweight="bold")
    fig.text(0.045, 0.60, "From Finite Elements", color=INK_ON_DARK, fontsize=33, fontweight="bold")
    fig.text(0.045, 0.43, "to Neural PDE Solvers", color=INK_ON_DARK, fontsize=33, fontweight="bold")
    fig.text(0.045, 0.235, "FEM theory  ·  FEniCS practice  ·  machine learning  ·  "
             "physics-informed networks", color=INK2_ON_DARK, fontsize=12)
    fig.text(0.045, 0.105, "Weak forms   →   finite elements   →   neural networks   →   "
             "PDE solvers", color=(1, 1, 1, 0.62), fontsize=11)
    img = fig_to_image(fig)
    plt.close(fig)
    img.save(out, optimize=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "banner.png")

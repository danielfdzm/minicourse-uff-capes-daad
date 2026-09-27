"""Course roadmap: three chapter cards on a transparent canvas (works on light and dark pages)."""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from style import INK, SECONDARY, GRID, LIGHT, use_course_fonts

W, H, DPI = 9.0, 3.45, 200
CARD_W, GAP, Y0, CARD_H, HEAD_H = 2.62, 0.47, 0.08, 3.28, 0.66

CARDS = [
    dict(number="1", title="Foundations of FEM",
         groups=[("1.1  Theory", ["weak forms and well-posedness", "Galerkin, meshes, assembly",
                                  "error estimates and adaptivity", "time-dependent problems"]),
                 ("1.2  FEniCS practice", ["Poisson to Stokes, nonlinear PDEs", "nonlocal PDEs, optimal control"])],
         chips=["142 slides", "14 Python examples"]),
    dict(number="2", title="Fundamentals of\nmachine learning",
         groups=[(None, ["supervised regression and risk", "linear models and least squares",
                         "neural networks as function classes", "universal approximation",
                         "gradient descent, backpropagation", "regularisation, early stopping",
                         "width, depth, model choice", "validation, testing, leakage"])],
         chips=["80 slides"]),
    dict(number="3", title="Neural networks\nfor solving PDEs",
         groups=[(None, ["strong-form PINNs", "Deep Ritz and weak formulations", "boundary conditions by design",
                         "hybrid losses, inverse problems", "limits vs classical solvers"]),
                 ("Neural PDE Laboratory", ["13 executed experiments", "interactive 3D explorers"])],
         chips=["83 slides", "Jupyter notebook"]),
]


def card(ax, x, spec):
    body = FancyBboxPatch((x, Y0), CARD_W, CARD_H, boxstyle="square,pad=0",
                          fc="white", ec=GRID, lw=1.0, zorder=1)
    ax.add_patch(body)
    head = Rectangle((x, Y0 + CARD_H - HEAD_H), CARD_W, HEAD_H, fc=LIGHT, ec="none", zorder=2)
    head.set_clip_path(body)
    ax.add_patch(head)
    top = Y0 + CARD_H - HEAD_H / 2
    ax.text(x + 0.17, top, spec["number"], color=INK, fontsize=25, fontweight="bold", va="center", zorder=3)
    ax.text(x + 0.52, top, spec["title"], color=INK, fontsize=11.5, fontweight="bold", va="center",
            linespacing=1.05, zorder=3)
    y = Y0 + CARD_H - HEAD_H - 0.25
    for heading, items in spec["groups"]:
        if heading:
            ax.text(x + 0.17, y, heading, color=INK, fontsize=9.5, fontweight="bold", va="center", zorder=3)
            y -= 0.215
        for item in items:
            ax.text(x + 0.19, y, "·", color=SECONDARY,
                    fontsize=8.5, va="center", zorder=3)
            ax.text(x + 0.36, y, item, color=INK, fontsize=9, va="center", zorder=3)
            y -= 0.205
        y -= 0.07
    ax.text(x + 0.17, Y0 + 0.26, "  ·  ".join(spec["chips"]),
            color=SECONDARY, fontsize=8, va="center", zorder=3)


def main(out):
    use_course_fonts()
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.patch.set_alpha(0)
    x0 = (W - 3 * CARD_W - 2 * GAP) / 2
    for i, spec in enumerate(CARDS):
        x = x0 + i * (CARD_W + GAP)
        card(ax, x, spec)
        if i < 2:
            ax.add_patch(FancyArrowPatch((x + CARD_W + 0.07, Y0 + CARD_H / 2), (x + CARD_W + GAP - 0.07, Y0 + CARD_H / 2),
                                         arrowstyle="->", mutation_scale=10,
                                         color=SECONDARY, lw=0.8, zorder=4))
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roadmap.png")

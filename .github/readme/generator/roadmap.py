"""Course roadmap: three chapter cards on a transparent canvas (works on light and dark pages)."""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from style import NAVY, TEAL, ORANGE, GRAY, PALE, use_course_fonts

W, H, DPI = 9.0, 3.45, 200
CARD_W, GAP, Y0, CARD_H, HEAD_H = 2.62, 0.47, 0.08, 3.28, 0.66

CARDS = [
    dict(colour=NAVY, number="1", title="Foundations of FEM",
         groups=[("1.1  Theory", ["weak forms and well-posedness", "Galerkin, meshes, assembly",
                                  "error estimates and adaptivity", "time-dependent problems"]),
                 ("1.2  FEniCS practice", ["Poisson to Stokes, nonlinear PDEs", "nonlocal PDEs, optimal control"])],
         chips=["142 slides", "14 Python examples"]),
    dict(colour=TEAL, number="2", title="Fundamentals of\nmachine learning",
         groups=[(None, ["supervised regression and risk", "linear models and least squares",
                         "neural networks as function classes", "universal approximation",
                         "gradient descent, backpropagation", "regularisation, early stopping",
                         "width, depth, model choice", "validation, testing, leakage"])],
         chips=["80 slides"]),
    dict(colour=ORANGE, number="3", title="Neural networks\nfor solving PDEs",
         groups=[(None, ["strong-form PINNs", "Deep Ritz and weak formulations", "boundary conditions by design",
                         "hybrid losses, inverse problems", "limits vs classical solvers"]),
                 ("Neural PDE Laboratory", ["13 executed experiments", "interactive 3D explorers"])],
         chips=["83 slides", "Jupyter notebook"]),
]


def card(ax, x, spec):
    body = FancyBboxPatch((x, Y0), CARD_W, CARD_H, boxstyle="round,pad=0,rounding_size=0.09",
                          fc="white", ec="#CFD9E4", lw=1.0, zorder=1)
    ax.add_patch(body)
    head = Rectangle((x, Y0 + CARD_H - HEAD_H), CARD_W, HEAD_H, fc=spec["colour"], ec="none", zorder=2)
    head.set_clip_path(body)
    ax.add_patch(head)
    top = Y0 + CARD_H - HEAD_H / 2
    ax.text(x + 0.17, top, spec["number"], color="white", fontsize=25, fontweight="bold", va="center", zorder=3)
    ax.text(x + 0.52, top, spec["title"], color="white", fontsize=11.5, fontweight="bold", va="center",
            linespacing=1.05, zorder=3)
    y = Y0 + CARD_H - HEAD_H - 0.25
    for heading, items in spec["groups"]:
        if heading:
            ax.text(x + 0.17, y, heading, color=NAVY, fontsize=9.5, fontweight="bold", va="center", zorder=3)
            y -= 0.215
        for item in items:
            ax.text(x + 0.19, y, "▸", color=spec["colour"] if spec["colour"] != NAVY else TEAL,
                    fontsize=8.5, va="center", zorder=3)
            ax.text(x + 0.36, y, item, color=NAVY, fontsize=9, va="center", zorder=3)
            y -= 0.205
        y -= 0.07
    cx = x + 0.17
    renderer = ax.figure.canvas.get_renderer()
    for chip in spec["chips"]:
        label = ax.text(cx + 0.11, Y0 + 0.26, chip, color=NAVY, fontsize=8, va="center", zorder=3)
        box = label.get_window_extent(renderer).transformed(ax.transData.inverted())
        width = box.width + 0.22                      # measured text width plus padding
        ax.add_patch(FancyBboxPatch((cx, Y0 + 0.14), width, 0.24, boxstyle="round,pad=0,rounding_size=0.11",
                                    fc=PALE, ec="none", zorder=2))
        cx += width + 0.08
        assert cx - 0.08 <= x + CARD_W - 0.1, f"chip {chip!r} overflows its card"


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
                                         arrowstyle="-|>,head_length=6,head_width=4", mutation_scale=1.6,
                                         color=TEAL, lw=2.2, zorder=4))
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roadmap.png")

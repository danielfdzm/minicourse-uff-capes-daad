"""Chapter 3 animation: a PINN learns the 2D Poisson solution from the PDE residual alone."""
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from style import (ANIMATION_FIGSIZE, ANIMATION_DPI, PAPER, PLOT_BLUE, PLOT_ORANGE, INK, SECONDARY, MUTED, GRID, LABEL,
                   ANIMATION_SOLUTION_CMAP, ANIMATION_ERROR_CMAP, use_course_fonts, fig_to_image, save_gif, style_axes)

STEPS, LR, N_COLLOCATION, SEED = 3000, 2e-3, 1024, 7
LIGHT = np.array([-0.45, -0.55, 0.85]) / np.linalg.norm([-0.45, -0.55, 0.85])


def train():
    torch.manual_seed(SEED)
    dt = torch.float64
    net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).to(dt)

    def u_theta(xy):                      # hard boundary factor: u = 0 on the boundary by construction
        x, y = xy[:, :1], xy[:, 1:]
        return x * (1 - x) * y * (1 - y) * net(2 * xy - 1)

    def residual(xy):                     # normalised residual (-Laplace u - f) / (2 pi^2)
        xy = xy.clone().requires_grad_(True)
        u = u_theta(xy)
        g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, i].sum(), xy, create_graph=True)[0][:, i:i + 1] for i in range(2))
        f = 2 * np.pi ** 2 * torch.sin(np.pi * xy[:, :1]) * torch.sin(np.pi * xy[:, 1:])
        return (-lap - f) / (2 * np.pi ** 2)

    pts = torch.quasirandom.SobolEngine(2, scramble=True, seed=SEED).draw(N_COLLOCATION).to(dt)
    ticks = torch.linspace(0, 1, 41, dtype=dt)
    X, Y = torch.meshgrid(ticks, ticks, indexing="xy")
    grid = torch.stack([X.ravel(), Y.ravel()], 1)
    exact = (torch.sin(np.pi * grid[:, 0]) * torch.sin(np.pi * grid[:, 1])).numpy().reshape(X.shape)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    snaps = set(np.unique(np.round(np.logspace(0, np.log10(STEPS), 56)).astype(int)))
    history, frames = [], []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = torch.mean(residual(pts) ** 2)
        loss.backward()
        opt.step()
        if step in snaps:
            with torch.no_grad():
                pred = u_theta(grid).numpy().reshape(X.shape)
            rel = np.linalg.norm(pred - exact) / np.linalg.norm(exact)
            history.append((step, loss.item() ** 0.5, rel))
            frames.append((step, pred, len(history)))
    return dict(X=X.numpy(), Y=Y.numpy(), exact=exact, history=np.array(history)), frames


def surface_polys(X, Y, Z):
    P = np.stack([X, Y, Z], -1)
    a, b, c, d = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
    tris = np.concatenate([np.stack([a, b, c], -2).reshape(-1, 3, 3), np.stack([a, c, d], -2).reshape(-1, 3, 3)])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
    n[n[:, 2] < 0] *= -1
    shade = 0.85 + 0.15 * np.clip(n @ LIGHT, 0, 1)
    colours = ANIMATION_SOLUTION_CMAP(np.clip(tris[:, :, 2].mean(1), 0, 1))[:, :3] * shade[:, None]
    return tris, np.clip(colours, 0, 1)


def draw_frame(data, snap, azim):
    step, pred, upto = snap
    hist = data["history"][:upto]
    fig = plt.figure(figsize=ANIMATION_FIGSIZE, dpi=ANIMATION_DPI, facecolor=PAPER)
    fig.text(0.045, 0.905, "CHAPTER 3  ·  NEURAL NETWORKS FOR PDEs", color=LABEL, fontsize=10, fontweight="bold")
    fig.text(0.045, 0.835, r"Physics-informed solution of $-\Delta u = f$", color=INK,
             fontsize=16, fontweight="normal")
    fig.text(0.045, 0.785, f"No solution data: the loss is the PDE residual at {N_COLLOCATION:,} points, "
             "and the boundary values are built into the network", color=SECONDARY, fontsize=10)

    # A: the network's current solution
    ax = fig.add_axes([-0.045, -0.05, 0.52, 0.86], projection="3d", facecolor=PAPER)
    ax.set_axis_off()
    tris, colours = surface_polys(data["X"], data["Y"], pred)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=colours, edgecolors=(1, 1, 1, 0.10), linewidths=0.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1.05)
    ax.set_box_aspect((1, 1, 0.62))
    ax.view_init(elev=27, azim=azim)
    fig.text(0.045, 0.705, r"network solution $u_\theta(x,y)$", color=SECONDARY, fontsize=10)

    # B: pointwise error (magnitude, log scale, perceptually ordered colour ramp)
    ex = fig.add_axes([0.47, 0.25, 0.19, 0.32])
    err = np.abs(pred - data["exact"]) + 1e-12
    im = ex.imshow(err, origin="lower", extent=(0, 1, 0, 1), cmap=ANIMATION_ERROR_CMAP,
                   norm=LogNorm(1e-5, 1.0), interpolation="bilinear")
    ex.set_xticks([]); ex.set_yticks([])
    for sp in ex.spines.values():
        sp.set_color(GRID)
    cb = fig.colorbar(im, ax=ex, fraction=0.05, pad=0.03, ticks=[1e-5, 1e-3, 1e-1])
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.text(0.47, 0.605, r"Error $|u_\theta - u|$", color=INK, fontsize=10.5, fontweight="bold")

    # C: training history - both quantities are dimensionless, so they share one log axis
    cx = fig.add_axes([0.795, 0.25, 0.17, 0.32])
    style_axes(cx)
    cx.set_xscale("log"); cx.set_yscale("log")
    cx.set_xlim(1, STEPS * 1.3); cx.set_ylim(3e-5, 2.0)
    cx.set_xticks([1, 10, 100, 1000])
    s, res, rel = hist[:, 0], hist[:, 1], hist[:, 2]
    cx.plot(s, res, color=PLOT_BLUE, linewidth=2, zorder=3)
    cx.plot(s, rel, color=PLOT_ORANGE, linestyle="--", linewidth=2, zorder=3)
    for series, colour in ((res, PLOT_BLUE), (rel, PLOT_ORANGE)):
        cx.scatter(s[-1:], series[-1:], s=40, color=colour, edgecolors=PAPER, linewidths=1.6, zorder=4)
    cx.set_xlabel("training step", color=MUTED, fontsize=9, labelpad=2)
    fig.text(0.795, 0.605, "Training", color=INK, fontsize=10.5, fontweight="bold")
    # legend (two series), set as text tokens beside colour keys
    for i, (label, colour) in enumerate((("PDE residual (loss)", PLOT_BLUE), (r"relative $L^2$ error", PLOT_ORANGE))):
        y = 0.105 - 0.05 * i
        fig.add_artist(plt.Line2D([0.795, 0.82], [y + 0.012, y + 0.012], color=colour, linewidth=2.2, linestyle="--" if i else "-",
                                  transform=fig.transFigure))
        fig.text(0.827, y, label, color=SECONDARY, fontsize=8.5)
    fig.text(0.47, 0.105, f"step {step:,}", color=INK, fontsize=11)
    fig.text(0.47, 0.055, rf"relative $L^2$ error {rel[-1]:.1e}".replace("e-0", "e-"), color=SECONDARY,
             fontsize=10)
    img = fig_to_image(fig, transparent=True)
    plt.close(fig)
    return img


def main(out):
    use_course_fonts()
    data, snaps = train()
    frames = [draw_frame(data, snap, -60) for snap in snaps]
    durations = [220] * len(frames)
    durations[0] = 1800
    durations[-1] = 4000
    save_gif(frames, out, durations)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pinn_training.gif")

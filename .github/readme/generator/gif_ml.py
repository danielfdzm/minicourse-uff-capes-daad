"""Chapter 2 animation: a small network fits noisy samples; train and validation loss diverge."""
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import (PAPER, BLUE, RUST, INK, SECONDARY, MUTED, LABEL,
                   use_course_fonts, fig_to_image, save_gif, style_axes)

SEED, SIGMA, N_TRAIN, N_VAL, WIDTH, LR, STEPS = 3, 0.15, 40, 20, 48, 5e-3, 6000
SIZE = (800, 450)


def target(x):
    return 0.8 * np.sin(2 * np.pi * x) + 0.35 * np.cos(5 * np.pi * x)


def train():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, N_TRAIN + N_VAL)
    y = target(x) + SIGMA * rng.standard_normal(x.size)
    xt, yt, xv, yv = (torch.tensor(a, dtype=torch.float64)[:, None]
                      for a in (x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:], y[N_TRAIN:]))
    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(1, WIDTH), torch.nn.Tanh(),
                              torch.nn.Linear(WIDTH, WIDTH), torch.nn.Tanh(),
                              torch.nn.Linear(WIDTH, 1)).double()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    grid = torch.linspace(0, 1, 400, dtype=torch.float64)[:, None]
    snaps = set(np.unique(np.round(np.logspace(0, np.log10(STEPS), 58)).astype(int)))
    history, frames = [], []
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        loss = torch.mean((net(2 * xt - 1) - yt) ** 2)       # inputs scaled to [-1, 1]
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = torch.mean((net(2 * xv - 1) - yv) ** 2).item()
        history.append((step, loss.item(), val))
        if step in snaps:
            with torch.no_grad():
                pred = net(2 * grid - 1).squeeze(1).numpy()
            frames.append((step, pred, len(history)))
    data = dict(xt=x[:N_TRAIN], yt=y[:N_TRAIN], xv=x[N_TRAIN:], yv=y[N_TRAIN:],
                grid=grid.squeeze(1).numpy(), history=np.array(history))
    return data, frames


def draw_frame(data, snap):
    step, pred, upto = snap
    hist = data["history"][:upto]
    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor=PAPER)
    fig.text(0.045, 0.905, "CHAPTER 2  ·  MACHINE LEARNING", color=LABEL, fontsize=10, fontweight="bold")
    fig.text(0.045, 0.835, "Neural network regression on noisy samples",
             color=INK, fontsize=16, fontweight="normal")
    fig.text(0.045, 0.785, f"1–{WIDTH}–{WIDTH}–1 tanh network, {N_TRAIN} training samples, "
             "mean-squared error, Adam", color=SECONDARY, fontsize=10)

    # fit panel
    ax = fig.add_axes([0.075, 0.13, 0.50, 0.55])
    style_axes(ax)
    ax.plot(data["grid"], target(data["grid"]), color=MUTED, linewidth=1.4, linestyle="--", zorder=1)
    ax.scatter(data["xt"], data["yt"], s=30, color=BLUE, edgecolors=PAPER, linewidths=1.4, zorder=3)
    ax.scatter(data["xv"], data["yv"], s=30, marker="s", color=RUST, edgecolors=PAPER, linewidths=1.4, zorder=3)
    ax.plot(data["grid"], pred, color=INK, linewidth=2.2, solid_capstyle="round", zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel("x", color=MUTED, fontsize=9, labelpad=2)
    handles = [Line2D([], [], color=INK, lw=2.2, label="network"),
               Line2D([], [], color=MUTED, lw=1.4, ls="--", label="hidden function"),
               Line2D([], [], color=BLUE, marker="o", lw=0, ms=6, mec=PAPER, label="training data"),
               Line2D([], [], color=RUST, marker="s", lw=0, ms=6, mec=PAPER, label="validation data")]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False,
                    fontsize=8.5, handlelength=1.6, columnspacing=1.2, labelcolor=SECONDARY)

    # loss panel: two series on one axis (same units)
    lx = fig.add_axes([0.665, 0.13, 0.30, 0.55])
    style_axes(lx)
    lx.set_xscale("log"); lx.set_yscale("log")
    lx.set_xlim(1, 1.2e5); lx.set_ylim(2.5e-3, 1.5)          # room for the end labels
    lx.set_xticks([1, 10, 100, 1000, 10000])
    lx.axhline(SIGMA ** 2, color=MUTED, linewidth=1.0)
    lx.text(1.3, SIGMA ** 2 * 1.12, "noise level σ²", color=MUTED, fontsize=8.5, va="bottom")
    s, tr, va = hist[:, 0], hist[:, 1], hist[:, 2]
    lx.plot(s, tr, color=BLUE, linewidth=2, zorder=3)
    lx.plot(s, va, color=RUST, linestyle="--", linewidth=2, zorder=3)
    lx.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="training"),
                       Line2D([], [], color=RUST, lw=2, ls="--", label="validation")],
              loc="upper right", frameon=False, fontsize=8.5, handlelength=1.4, labelcolor=SECONDARY)
    # direct end labels only once the curves have separated (never stacked on each other)
    separated = abs(np.log10(tr[-1]) - np.log10(va[-1])) > 0.18
    for series, colour, name in ((tr, BLUE, "training"), (va, RUST, "validation")):
        lx.scatter(s[-1:], series[-1:], s=44, color=colour, edgecolors=PAPER, linewidths=1.6, zorder=4)
        if separated:
            lx.annotate(name, (s[-1], series[-1]), xytext=(7, -3), textcoords="offset points",
                        color=SECONDARY, fontsize=8.5)
    best = int(np.argmin(va))
    if va[-1] > 1.6 * va[best]:                       # overfitting is visible: mark the checkpoint
        lx.scatter([s[best]], [va[best]], s=90, facecolors="none", edgecolors=INK, linewidths=1.6, zorder=5)
        lx.annotate("best checkpoint", (s[best], va[best]), xytext=(-8, -16), textcoords="offset points",
                    color=INK, fontsize=8.5, ha="right")
    lx.set_xlabel("training step", color=MUTED, fontsize=9, labelpad=2)
    fig.text(0.665, 0.715, "Mean-squared error", color=INK, fontsize=10.5, fontweight="bold")
    fig.text(0.045, 0.035, f"step {step:,}", color=INK, fontsize=11)
    img = fig_to_image(fig, SIZE)
    plt.close(fig)
    return img


def main(out):
    use_course_fonts()
    data, snaps = train()
    frames = [draw_frame(data, sn) for sn in snaps]
    durations = [220] * len(frames)
    durations[0] = 1800
    durations[-1] = 4000                               # hold the final state
    save_gif(frames, out, durations)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml_training.gif")

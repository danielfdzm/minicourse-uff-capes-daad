#!/usr/bin/env python3
"""Build slide figures, numeric TeX macros, and source excerpts from saved results.

Run after the three experiment scripts. No network training takes place here.
The PDF figures retain vector text and lines; PNG previews are also exported.
"""
from pathlib import Path
import json
import os
import textwrap
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np

FIGURES = ROOT / "figures"
SNIPPETS = ROOT / "snippets"
NAVY = "#193C55"
TEAL = "#117479"
ORANGE = "#CE6F24"
GREY = "#647786"
LIGHT = "#E3EAED"
PALE = "#F4F6F7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": GREY, "ytick.color": GREY,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.edgecolor": "#BAC8CF", "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": PALE, "figure.facecolor": "white",
    "grid.color": LIGHT, "grid.linewidth": 0.7,
    "lines.linewidth": 2.5, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 10.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
})
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "course_field", ["#F0F7F5", "#90C7C2", TEAL, NAVY])
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "course_error", ["#FFF9EF", "#F7D49C", ORANGE, "#71321E"])


def load(name):
    return (np.load(ROOT / "results" / f"{name}.npz"),
            json.loads((ROOT / "results" / f"{name}.json").read_text()))


def pair():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.045, h_pad=0.05, wspace=0.10)
    for ax in axes:
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    return fig, axes


def title(ax, text):
    ax.set_title(text, loc="left", pad=14, color=NAVY)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def forward_plot(d):
    fig, (left, right) = pair()
    title(left, "A   A learned solution")
    left.plot(d["x"], d["hard_pinn_u"], color=TEAL, label="PINN", lw=3.2)
    left.plot(d["x"], d["exact_u"], color=NAVY, ls=(0, (4, 3)),
              label="Exact", lw=1.7)
    left.set(xlabel="$x$", ylabel="$u(x)$", xlim=(0, 1), ylim=(-0.03, 1.14))
    left.legend(loc="upper right")
    title(right, "B   Error at evaluation points")
    error = np.abs(d["hard_pinn_u"] - d["exact_u"])
    # A visible scale factor preserves the original values without tiny tick labels.
    right.fill_between(d["x"], error * 1e6, color=ORANGE, alpha=0.16)
    right.plot(d["x"], error * 1e6, color=ORANGE)
    right.set(xlabel="$x$", ylabel=r"$|u_\theta-u|\;[\times 10^{-6}]$",
              xlim=(0, 1), ylim=(0, None))
    save(fig, "poisson_1d")


def field_plot(d):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.8), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.02, wspace=0.05)
    for ax, data, label in zip(axes, [d["u_true"], d["u_pred"], d["abs_error"] * 1e6],
                              ["A   Exact field", "B   PINN field", "C   Absolute error"]):
        is_error = ax is axes[2]
        im = ax.imshow(data, origin="lower", extent=[0, 1, 0, 1],
                       cmap=ERROR_CMAP if is_error else FIELD_CMAP,
                       vmin=0, vmax=float(data.max()) if is_error else 1,
                       interpolation="nearest", aspect="equal")
        title(ax, label)
        ax.set(xlabel="$x$", ylabel="$y$", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        bar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.18,
                           ticks=[0, 3, 6, 9] if is_error else [0, 0.5, 1])
        bar.outline.set_visible(False)
        bar.set_label(r"$|u_\theta-u|\;[\times 10^{-6}]$" if is_error else
                      (r"$u_\theta(x,y)$" if ax is axes[1] else "$u(x,y)$"))
    save(fig, "poisson_2d")


def boundary_plot(d):
    fig, (left, right) = pair()
    title(left, "A   Endpoint error")
    title(right, "B   Interior solution error")
    left.loglog(d["soft_lambda"], d["soft_boundary_max_abs"], "o-", color=ORANGE,
                markersize=6, markeredgecolor="white", label="Soft boundary")
    right.loglog(d["soft_lambda"], d["soft_relative_l2"], "o-", color=TEAL,
                 markersize=6, markeredgecolor="white", label="Soft boundary")
    right.axhline(float(d["hard_adam_relative_l2"]), color=NAVY, ls="--", lw=1.6,
                  label="Hard boundary")
    left.text(0.04, 0.94, "Hard boundary: exactly zero", transform=left.transAxes,
              ha="left", va="top", fontsize=10.5, color=NAVY,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
    left.set_ylabel(r"$\max(|u_\theta(0)|,|u_\theta(1)|)$")
    right.set_ylabel("Relative $L^2$ error")
    right.legend(loc="upper left")
    for ax in [left, right]:
        ax.set_xlabel(r"Boundary weight $\lambda_b$")
        ax.set_xticks([0.01, 1, 100, 1000])
        ax.set_xlim(0.006, 1700)
    save(fig, "boundary_sweep")


def ritz_plot(d):
    fig, axes = pair()
    for ax, metric, label in zip(axes, ["relative_l2", "relative_h1_seminorm"],
                                  ["A   Solution error", "B   Gradient error"]):
        title(ax, label)
        for model, color, name, marker in [("hard_pinn", TEAL, "PINN", "o"),
                                            ("ritz", ORANGE, "Deep Ritz", "s")]:
            mask = d[f"{model}_history_stage"] != "lbfgs"
            ax.semilogy(d[f"{model}_history_step"][mask],
                        d[f"{model}_history_{metric}"][mask], color=color,
                        label=name, marker=marker, markevery=4, ms=4.5)
        ax.set(xlabel="Adam update", xlim=(0, 2000))
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Relative $L^2$ error")
    axes[1].set_ylabel("Relative $H^1$ seminorm error")
    save(fig, "ritz_comparison")


def inverse_plots(d):
    fig, (left, right) = pair()
    title(left, "A   Reconstruct from 12 sensors")
    left.plot(d["x"], d["u_hybrid"], color=TEAL, label="Hybrid PINN", lw=3)
    left.plot(d["x"], d["u_true"], color=NAVY, ls=(0, (4, 3)), label="Exact", lw=1.6)
    left.errorbar(d["x_data"], d["y_data"], yerr=float(d["noise_sigma"]),
                   fmt="o", color=ORANGE, ms=4.5, capsize=2.5,
                   markeredgecolor="white", elinewidth=1.3, label=r"Sensors ($\pm\sigma$)")
    left.set(xlabel="$x$", ylabel="$u(x)$", xlim=(0, 1), ylim=(-0.05, 1.95))
    left.legend(loc="upper left", fontsize=9.5, ncol=2, columnspacing=0.7,
                handlelength=1.5)
    title(right, "B   Identify the coefficient")
    for name, color, label in [("hybrid", TEAL, "PDE + data"),
                                 ("physics_only", ORANGE, "PDE only")]:
        mask = d[f"{name}_phase"] == 0
        right.plot(d[f"{name}_step"][mask], d[f"{name}_k"][mask], color=color, label=label)
    right.axhline(float(d["true_k"]), color=NAVY, ls="--", lw=1.5, label=r"True $\kappa=0.7$")
    right.set(xlabel="Adam update", ylabel=r"Estimate $\widehat\kappa$", xlim=(0, 4000), ylim=(0.5, 1.65))
    right.legend(loc="center right", fontsize=10)
    save(fig, "inverse_recovery")

    fig, (left, right) = pair()
    title(left, "A   All curves satisfy the PDE")
    colors = [GREY, TEAL, ORANGE, "#917A9F"]
    for k, u, color in zip(d["family_example_k"], d["family_example_u"], colors):
        left.plot(d["x"], u, color=color, label=rf"$\kappa={k:g}$",
                  lw=2.7 if k == 0.7 else 1.8)
    left.scatter(d["x_data"], d["y_data"], color=NAVY, s=17, zorder=5,
                 edgecolor="white", lw=0.5)
    left.set(xlabel="$x$", ylabel=r"$u_\kappa(x)$", xlim=(0, 1), ylim=(0, 2.65))
    left.legend(loc="upper center", ncol=2, fontsize=10, columnspacing=0.8)
    title(right, "B   Sensors distinguish coefficients")
    right.semilogy(d["family_k"], d["family_sensor_mse"], color=TEAL, label="Sensor MSE")
    right.axvline(float(d["true_k"]), color=NAVY, ls="--", lw=1.5, label=r"True $\kappa$")
    right.axhline(float(d["noise_sigma"])**2, color=ORANGE, ls=":", lw=1.8,
                   label=r"Noise variance $\sigma^2$")
    right.set(xlabel=r"Candidate $\kappa$", ylabel="Mean squared sensor mismatch",
               xlim=(d["family_k"].min(), d["family_k"].max()))
    right.legend(loc="upper right", fontsize=10)
    save(fig, "inverse_identifiability")


def classical_plot(d):
    fig, (left, right) = pair()
    title(left, "A   Refinement reduces PDE error")
    left.loglog(d["fd_node_count"], d["fd_relative_l2"], "o-", color=TEAL,
                ms=4.5, label="Finite differences")
    left.loglog(d["spectral_node_count"], d["spectral_relative_l2"], "s-",
                color=NAVY, ms=4.5, label="Chebyshev spectral")
    left.axhline(float(d["pinn_relative_l2"]), color=ORANGE, ls="--", lw=1.6,
                  label="PINN: existing run")
    left.axhline(float(d["float64_eps"]), color=GREY, ls=":", lw=1.5)
    left.text(0.97, 0.02, r"float64 $\varepsilon$", transform=left.transAxes,
               ha="right", va="bottom", fontsize=10, color=GREY)
    left.set(xlabel="Nodes (classical methods)", ylabel="Relative $L^2$ error",
               xlim=(7, 1600), ylim=(1e-16, 2e3), yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1])
    left.legend(loc="upper right", fontsize=9.5)
    title(right, "B   A solved system can be coarse")
    values = [float(d["fd_nodal_residual_relative_inf"][0]),
              float(d["fd_relative_l2"][0])]
    for x, value, color in zip([0, 1], values, [NAVY, ORANGE]):
        right.vlines(x, 1e-16, value, color=color, lw=9, alpha=0.22)
        right.scatter(x, value, s=65, color=color, zorder=4)
        exponent = int(np.floor(np.log10(value)))
        label = rf"${value / 10.**exponent:.2f}\times10^{{{exponent}}}$"
        right.annotate(label, (x, value), xytext=(0, 12), textcoords="offset points",
                         ha="center", color=color, fontsize=11)
    right.set_yscale("log")
    right.set(xlim=(-0.55, 1.55), ylim=(1e-16, 1),
               xticks=[0, 1], xticklabels=["Relative discrete\nresidual ($\\infty$ norm)",
                                           "Relative solution\nerror ($L^2$ norm)"],
               ylabel="Dimensionless diagnostic", yticks=[1e-16, 1e-12, 1e-8, 1e-4, 1])
    right.tick_params(axis="x", labelsize=10)
    right.text(0.05, 0.97, "Finite differences, 17 nodes", transform=right.transAxes,
                va="top", fontsize=10)
    save(fig, "classical_accuracy")


def classical_metrics(d):
    best = int(np.argmin(d["spectral_relative_l2"]))
    content = "% Generated from actual classical_accuracy.npz results.\n"
    content += (rf"\newcommand{{\ClassicalSpectralError}}{{{scientific(d['spectral_relative_l2'][best])}}}"
                "\n" + rf"\newcommand{{\ClassicalBestDegree}}{{{int(d['spectral_degree'][best])}}}" + "\n")
    (ROOT / "metrics_classical.tex").write_text(content)


def snippet(script, marker):
    source = (ROOT / script).read_text()
    return textwrap.dedent(source.split(f"# BEGIN {marker}\n", 1)[1]
                           .split(f"# END {marker}", 1)[0]).strip() + "\n"


def make_snippets():
    forward = (ROOT / "forward_1d.py").read_text()
    setup = textwrap.dedent("    initial = nn.Sequential(" + forward.split(
        "    initial = nn.Sequential(", 1)[1].split("    hard = train", 1)[0])
    setup = setup.replace("grid = np.linspace(0.0, 1.0, 1001)\n", "")
    setup = setup.replace(
        "nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)",
        "nn.Linear(1, 32), nn.Tanh(),\n"
        "    nn.Linear(32, 32), nn.Tanh(),\n"
        "    nn.Linear(32, 1)")
    setup = "# Imports: numpy as np, torch, and torch.nn as nn\n" + setup
    (SNIPPETS / "forward_setup.py").write_text(setup.strip() + "\n")
    loss = ("def derivative(y, x):\n"
            "    return torch.autograd.grad(\n"
            "        y, x, torch.ones_like(y), create_graph=True)[0]\n\n")
    loss += snippet("forward_1d.py", "PINN_LOSS")
    loss += ("optimizer = torch.optim.Adam(hard_model.parameters(), lr=1e-3)\n"
             "optimizer.zero_grad()\n"
             "loss = pinn_loss(hard_model, x, weights)\n"
             "loss.backward()\n"
             "optimizer.step()\n")
    (SNIPPETS / "forward_loss.py").write_text(loss)
    (SNIPPETS / "boundary_loss.py").write_text(snippet("forward_1d.py", "SOFT_LOSS"))
    (SNIPPETS / "ritz_loss.py").write_text(snippet("forward_1d.py", "RITZ_LOSS"))
    (SNIPPETS / "poisson_2d_loss.py").write_text(snippet("poisson_2d.py", "LAPLACIAN_SNIPPET"))
    inv = ("# Inside the model; raw_k is a trainable nn.Parameter.\n"
           "@property\n"
           "def k(self):\n"
           "    return F.softplus(self.raw_k) + 1e-6\n\n"
           "# r = (-k * u_xx - f) / pi**2; data_weight = 5.0\n")
    inv += snippet("inverse_1d.py", "inverse_loss")
    inv += "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
    (SNIPPETS / "inverse_loss.py").write_text(inv)
    (SNIPPETS / "reproduce.sh").write_text(
        "python3 -m pip install -r experiments/requirements.txt\n"
        "python3 experiments/forward_1d.py\n"
        "python3 experiments/poisson_2d.py\n"
        "python3 experiments/inverse_1d.py\n"
        "python3 experiments/classical_accuracy.py\n"
        "python3 experiments/make_figures.py\n"
        "latexmk -pdf 3_neural_networks_for_pdes.tex\n")


def scientific(value):
    mantissa, power = f"{float(value):.2e}".split("e")
    return rf"\ensuremath{{{mantissa}\times10^{{{int(power)}}}}}"


def metrics(f, fm, tm, im):
    main = fm["main_models"]
    inv = im["metrics"]["hybrid"]
    macros = {
        "ForwardLtwo": scientific(main["hard_pinn"]["final"]["relative_l2"]),
        "ForwardHone": scientific(main["hard_pinn"]["final"]["relative_h1_seminorm"]),
        "ForwardMaxError": scientific(np.max(np.abs(f["hard_pinn_u"]-f["exact_u"]))),
        "RitzLtwo": scientific(main["ritz"]["final"]["relative_l2"]),
        "RitzHone": scientific(main["ritz"]["final"]["relative_h1_seminorm"]),
        "TwoDLtwo": scientific(tm["metrics"]["relative_l2"]),
        "TwoDMaxError": scientific(tm["metrics"]["max_abs_error"]),
        "InverseKappa": f'{inv["inferred_k"]:.6f}',
        "InverseKappaError": f'{inv["k_relative_error_percent"]:.3f}\\%',
        "InverseLtwo": scientific(inv["held_out_relative_l2"]),
        "BoundarySteps": str(fm["config"]["adam_steps"]),
        "ForwardProtocol": "128 Gauss--Legendre nodes; 2,000 Adam updates + 300 L-BFGS iterations",
        "RitzProtocol": "128 quadrature nodes; 2,000 Adam updates + 300 L-BFGS iterations",
        "TwoDProtocol": r"Seed 7; $2$--$32$--$32$--$32$--$1$ tanh network; 1,024 Sobol points; 2,000 Adam + up to 600 L-BFGS iterations",
        "InverseDataProtocol": r"12 sensors with independent Gaussian noise, $\sigma=0.0143$ (1\% of peak amplitude), seed 23",
        "InverseTrainProtocol": r"128 interior points; $1$--$32$--$32$--$1$ tanh network; 4,000 Adam + up to 200 L-BFGS iterations",
    }
    content = "% Generated from experiments/results by make_figures.py.\n"
    content += "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items())
    (ROOT / "metrics.tex").write_text(content + "\n")


def main():
    FIGURES.mkdir(exist_ok=True)
    SNIPPETS.mkdir(exist_ok=True)
    f, fm = load("forward_1d")
    t, tm = load("poisson_2d")
    i, im = load("inverse_1d")
    c, cm = load("classical_accuracy")
    forward_plot(f)
    field_plot(t)
    boundary_plot(f)
    ritz_plot(f)
    inverse_plots(i)
    classical_plot(c)
    classical_metrics(c)
    make_snippets()
    metrics(f, fm, tm, im)
    # Analytic illustrations run separately so their plotting settings stay local.
    for script in ["wave_data.py", "counterexample_data.py", "continuous_residual.py"]:
        subprocess.run([sys.executable, str(ROOT / script)], check=True)
    print("Created 10 PDF figures, 10 PNG previews, source excerpts, and metric macros.")


if __name__ == "__main__":
    main()

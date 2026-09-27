"""Notebook cells: sparse-sensor inverse diffusion and identifiability."""

CELLS = [
    {
        "cell_type": "markdown",
        "source": r"""## Inverse diffusion: where you measure matters

Use **ten noisy observations** to estimate a positive diffusion coefficient and the corresponding solution:

\[
-\kappa u''=\pi^2\sin(\pi x),\qquad u(0)=u(1)=0,
\qquad u(x)=\frac{\sin(\pi x)}{\kappa},\quad \kappa_{\rm true}=0.7.
\]

Compare well-spread interior sensors with sensors clustered near the boundaries. Both layouts use the same number of sensors, the same absolute noise level, and the same noise vector. A small neural network learns $u_\theta$ and $\kappa$ jointly, using $u_\theta=x(1-x)N_\theta$ to enforce both boundary values. We also fit the exact solution formula by least squares. Comparing the two fits helps separate training error from data uncertainty.

**Before running:** the boundaries fix $u=0$ for *every* coefficient. Measurements close to them carry less information about $\kappa$, even though the data still contain noise.""",
    },
    {
        "cell_type": "code",
        "source": r"""# One seed, one noise realization, two measurement designs.
inv_rng = np.random.default_rng(23)
inv_kappa_true, inv_sigma = 0.7, 0.02 / 0.7
inv_count = 10
inv_noise = inv_rng.normal(0.0, inv_sigma, inv_count)
inv_layouts = {
    "Interior sensors": np.linspace(0.12, 0.88, inv_count),
    "Boundary clusters": np.r_[np.linspace(0.01, 0.06, 5),
                               np.linspace(0.94, 0.99, 5)],
}
inv_grid = np.linspace(0.0, 1.0, 401)
inv_sine = np.sin(np.pi * inv_grid)
inv_results = {}
for inv_label, inv_x in inv_layouts.items():
    inv_s = np.sin(np.pi * inv_x)
    inv_y = inv_s / inv_kappa_true + inv_noise
    # Let c = 1/kappa: this is a one-parameter linear least-squares problem.
    inv_c = np.dot(inv_s, inv_y) / np.dot(inv_s, inv_s)
    inv_se = inv_sigma / np.sqrt(np.dot(inv_s, inv_s))
    inv_c_bounds = inv_c + np.array([-1.0, 1.0]) * 1.95996398454 * inv_se
    assert inv_c_bounds[0] > 0, "Increase information before inverting this interval."
    inv_results[inv_label] = dict(
        x=inv_x, y=inv_y, c=inv_c, c_bounds=inv_c_bounds,
        kappa_ls=1.0 / inv_c, kappa_ci=1.0 / inv_c_bounds[::-1],
        information=np.dot(inv_s, inv_s) / (inv_sigma**2 * inv_kappa_true**4),
    )
print("Equal data budgets: 10 sensors each; noise SD = %.4f" % inv_sigma)
for inv_label, inv_result in inv_results.items():
    print(f"{inv_label:19s}  kappa = {inv_result['kappa_ls']:.4f}  "
          f"95% interval = {inv_result['kappa_ci'].round(4)}")
print("Information ratio (interior / boundary): %.1f" % (
    inv_results["Interior sensors"]["information"] /
    inv_results["Boundary clusters"]["information"]))""",
    },
    {
        "cell_type": "code",
        "source": r"""# Joint neural / coefficient training, starting from identical weights.
# Physics is normalized by pi^2; the data term receives weight 5.
inv_dtype = torch.float64

class inv_PINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 24), torch.nn.Tanh(),
            torch.nn.Linear(24, 24), torch.nn.Tanh(),
            torch.nn.Linear(24, 1))
        self.log_kappa = torch.nn.Parameter(torch.tensor(np.log(1.1), dtype=inv_dtype))

    @property
    def kappa(self):
        return self.log_kappa.exp()

    def forward(self, x):
        return x * (1.0 - x) * self.net(x)

def inv_train(inv_result):
    torch.manual_seed(23)
    inv_model = inv_PINN().to(dtype=inv_dtype)
    inv_xr = torch.linspace(0, 1, 66, dtype=inv_dtype)[1:-1, None].requires_grad_()
    inv_xd = torch.tensor(inv_result["x"][:, None], dtype=inv_dtype)
    inv_yd = torch.tensor(inv_result["y"][:, None], dtype=inv_dtype)

    def inv_loss():
        inv_u = inv_model(inv_xr)
        inv_du = torch.autograd.grad(inv_u, inv_xr, torch.ones_like(inv_u),
                                     create_graph=True)[0]
        inv_d2u = torch.autograd.grad(inv_du, inv_xr, torch.ones_like(inv_du),
                                      create_graph=True)[0]
        inv_residual = -inv_model.kappa * inv_d2u / np.pi**2 - torch.sin(np.pi * inv_xr)
        return inv_residual.square().mean() + 5.0 * (inv_model(inv_xd) - inv_yd).square().mean()

    inv_history = []
    inv_optimizer = torch.optim.Adam(inv_model.parameters(), lr=0.003)
    for inv_step in range(1000):
        inv_optimizer.zero_grad(set_to_none=True)
        inv_xr.grad = None
        inv_value = inv_loss()
        inv_value.backward()
        inv_optimizer.step()
        if inv_step % 20 == 0:
            inv_history.append((inv_step + 1, inv_model.kappa.item()))
    inv_optimizer = torch.optim.LBFGS(
        inv_model.parameters(), max_iter=140, history_size=30,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def inv_closure():
        inv_optimizer.zero_grad(set_to_none=True)
        inv_xr.grad = None
        inv_value = inv_loss()
        inv_value.backward()
        return inv_value
    inv_optimizer.step(inv_closure)
    with torch.no_grad():
        inv_prediction = inv_model(torch.tensor(inv_grid[:, None], dtype=inv_dtype)).numpy().ravel()
    return inv_model.kappa.item(), inv_prediction, np.asarray(inv_history)

inv_old_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    for inv_label, inv_result in inv_results.items():
        inv_k, inv_prediction, inv_history = inv_train(inv_result)
        inv_result.update(kappa_nn=inv_k, prediction=inv_prediction, history=inv_history)
        inv_error = np.linalg.norm(inv_prediction - inv_sine / inv_kappa_true) / np.linalg.norm(inv_sine / inv_kappa_true)
        print(f"{inv_label:19s}  neural kappa = {inv_k:.4f}  "
              f"reference kappa = {inv_result['kappa_ls']:.4f}  field error = {inv_error:.2%}")
finally:
    torch.set_num_threads(inv_old_threads)""",
    },
    {
        "cell_type": "code",
        "source": r"""inv_colors = [palette["teal"], palette["pink"]]
inv_fig, inv_axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
for inv_ax, (inv_label, inv_result), inv_color in zip(
        inv_axes[0], inv_results.items(), inv_colors):
    inv_ax.fill_between(inv_grid, inv_sine * inv_result["c_bounds"][0],
                        inv_sine * inv_result["c_bounds"][1], color=inv_color,
                        alpha=0.18, label="95% exact-model band")
    inv_ax.plot(inv_grid, inv_sine / inv_kappa_true, color=palette["ink"], lw=2,
                ls="--", label="True field")
    inv_ax.plot(inv_grid, inv_result["prediction"], color=inv_color, lw=3,
                label="Trained PINN")
    inv_ax.errorbar(inv_result["x"], inv_result["y"], yerr=inv_sigma,
                    fmt="o", ms=6, capsize=3, color=inv_color,
                    mec="white", mew=0.8, label=r"Sensors $\pm\sigma$", zorder=5)
    inv_ax.set(title=inv_label, xlabel="$x$", ylabel="$u(x)$", xlim=(0, 1), ylim=(-0.1, 1.85))
    inv_ax.legend(loc="upper right", fontsize=8)

inv_kgrid = np.linspace(0.35, 1.25, 700)
for (inv_label, inv_result), inv_color in zip(inv_results.items(), inv_colors):
    inv_at_sensors = np.sin(np.pi * inv_result["x"])[None, :] / inv_kgrid[:, None]
    inv_nll = 0.5 * np.sum(((inv_at_sensors - inv_result["y"]) / inv_sigma)**2, axis=1)
    inv_best_nll = 0.5 * np.sum(((inv_result["c"] * np.sin(np.pi * inv_result["x"])
                                - inv_result["y"]) / inv_sigma)**2)
    inv_axes[1, 0].plot(inv_kgrid, inv_nll - inv_best_nll,
                        color=inv_color, lw=2.8, label=inv_label)
inv_axes[1, 0].axhline(1.95996398454**2 / 2, color=palette["muted"], ls=":", lw=1.5,
                      label="95% likelihood threshold")
inv_axes[1, 0].axvline(inv_kappa_true, color=palette["ink"], ls="--", lw=1)
inv_axes[1, 0].set(xlim=(0.48, 1.03), ylim=(0, 12), xlabel=r"Candidate $\kappa$",
                   ylabel="Excess negative log-likelihood", title="Wide valleys mean weaker information")
inv_axes[1, 0].legend(fontsize=8)

for inv_row, ((inv_label, inv_result), inv_color) in enumerate(zip(inv_results.items(), inv_colors)):
    inv_ci = inv_result["kappa_ci"]
    inv_axes[1, 1].plot(inv_ci, [inv_row, inv_row], lw=9, solid_capstyle="round", color=inv_color, alpha=0.3)
    inv_axes[1, 1].scatter(inv_result["kappa_ls"], inv_row, s=100, color=inv_color,
                          edgecolor="white", zorder=4, label="Exact-model fit" if inv_row == 0 else None)
    inv_axes[1, 1].scatter(inv_result["kappa_nn"], inv_row, marker="*", s=230,
                          color=palette["orange"], edgecolor=palette["ink"], linewidth=0.5,
                          zorder=5, label="Joint PINN fit" if inv_row == 0 else None)
inv_axes[1, 1].axvline(inv_kappa_true, color=palette["ink"], ls="--", lw=1.5, label="Truth")
inv_axes[1, 1].set(yticks=[0, 1], yticklabels=list(inv_results), ylim=(-0.6, 1.6),
                  xlabel=r"Recovered $\kappa$", title="Same budget, different uncertainty")
inv_axes[1, 1].legend(loc="upper right", fontsize=8)
inv_fig.suptitle("The measurement design changes the inverse problem", fontsize=20, weight="bold")
save_figure(inv_fig, "inverse_sensor_design")""",
    },
    {
        "cell_type": "markdown",
        "source": r"""### What can the observations tell us?

For independent Gaussian noise of known standard deviation $\sigma$, set $s_i=\sin(\pi x_i)$ and $c=1/\kappa$. Then

\[
\widehat c=\frac{\sum_i s_i y_i}{\sum_i s_i^2},\qquad
\operatorname{sd}(\widehat c)=\frac{\sigma}{\sqrt{\sum_i s_i^2}},\qquad
I(\kappa)=\frac{\sum_i\sin^2(\pi x_i)}{\sigma^2\kappa^4}.
\]

We obtain the plotted intervals from the **exact forward model** by transforming a Gaussian interval for $c$. They describe uncertainty in that fit, rather than in the neural prediction. The PINN can give a different estimate because its loss includes a sampled, weighted PDE residual and training stops after a fixed number of steps.

Now free the forcing amplitude too: $-\kappa u''=a\pi^2\sin(\pi x)$. The solution becomes $u=(a/\kappa)\sin(\pi x)$. Even perfect observations determine only the ratio $a/\kappa$. The second 3D plot shows the resulting valley. To identify both parameters, we need an additional kind of measurement.""",
    },
    {
        "cell_type": "code",
        "source": r"""# 3D geometry: the forward solution family and a two-parameter inverse valley.
inv_fig = plt.figure(figsize=(15.0, 6.4), constrained_layout=True)
inv_ax_family = inv_fig.add_subplot(121, projection="3d")
inv_ax_valley = inv_fig.add_subplot(122, projection="3d")
inv_ax_family.computed_zorder = False
inv_ax_valley.computed_zorder = False
inv_family_k = np.linspace(0.38, 1.35, 70)
inv_X, inv_K = np.meshgrid(np.linspace(0, 1, 100), inv_family_k)
inv_U = np.sin(np.pi * inv_X) / inv_K
inv_surface = inv_ax_family.plot_surface(inv_X, inv_K, inv_U, cmap="viridis",
                                          linewidth=0, antialiased=True, alpha=0.90, zorder=2)
inv_ax_family.plot(inv_grid, np.full_like(inv_grid, inv_kappa_true), inv_sine / inv_kappa_true,
                    color=palette["orange"], lw=4, label="True coefficient", zorder=8)
inv_ax_family.contour(inv_X, inv_K, inv_U, zdir="z", offset=-0.12, levels=10,
                       cmap="viridis", linewidths=0.7, alpha=0.6, zorder=1)
inv_ax_family.set(xlabel="$x$", ylabel=r"$\kappa$", zlabel=r"$u_\kappa(x)$", zlim=(-0.12, 2.8))
style_3d(inv_ax_family, title="Every slice solves the PDE")
inv_ax_family.view_init(elev=28, azim=-57)
inv_ax_family.legend(loc="upper left", fontsize=8)

inv_ref = inv_results["Interior sensors"]
inv_K2, inv_A = np.meshgrid(np.linspace(0.35, 1.3, 100), np.linspace(0.4, 1.9, 110))
inv_prediction_family = (inv_A / inv_K2)[..., None] * np.sin(np.pi * inv_ref["x"])
inv_loss_surface = 0.5 * np.sum(((inv_prediction_family - inv_ref["y"]) / inv_sigma)**2, axis=-1)
inv_loss_min = 0.5 * np.sum(((inv_ref["c"] * np.sin(np.pi * inv_ref["x"])
                             - inv_ref["y"]) / inv_sigma)**2)
inv_Z = np.sqrt(2.0 * np.maximum(inv_loss_surface - inv_loss_min, 0.0) / inv_count)
inv_surface = inv_ax_valley.plot_surface(inv_K2, inv_A, inv_Z, cmap="magma",
                                         linewidth=0, antialiased=True, alpha=0.88, zorder=2)
inv_ridge_k = np.linspace(max(0.35, 0.4 / inv_ref["c"]),
                          min(1.3, 1.9 / inv_ref["c"]), 150)
inv_ax_valley.plot(inv_ridge_k, inv_ridge_k * inv_ref["c"], np.zeros_like(inv_ridge_k),
                    color=palette["teal"], lw=4, label=r"Identical best fits: $a=\widehat c\kappa$", zorder=8)
inv_ax_valley.set(xlabel=r"$\kappa$", ylabel="Forcing amplitude $a$", zlabel="")
inv_ax_valley.text2D(0.98, 0.89, r"Height: $\sqrt{2\,\Delta\mathrm{NLL}/N_d}$",
                      ha="right", va="center", transform=inv_ax_valley.transAxes)
style_3d(inv_ax_valley, title="Two unknowns, one identifiable ratio")
inv_ax_valley.view_init(elev=32, azim=-57)
inv_ax_valley.legend(loc="upper left", fontsize=8)
inv_fig.suptitle("Explore the geometry of an inverse PDE", fontsize=20, weight="bold")
save_figure(inv_fig, "inverse_solution_family_and_valley")""",
    },
    {
        "cell_type": "markdown",
        "source": r"""**Try it yourself**

1. Move the boundary clusters toward $x=1/2$. Predict the information ratio before rerunning. For this parameter, sensors near the midpoint are informative. Useful placement depends on what you want to recover.
2. Increase the noise by a factor of two. How should the width of the interval for $c$ change? Does the neural fit remain inside the transformed interval for $\kappa$? Repeat with fresh noise to see how often the intervals contain the true value.
3. For the two-unknown experiment, imagine adding one calibrated flux observation $q=-\kappa u'$. It gives $q=-a\pi\cos(\pi x)$; away from $x=1/2$, this can identify $a$ separately and break the ratio ambiguity.
4. Reduce the physics weight or the training iterations. Compare the neural and exact-model estimates to see why data uncertainty and numerical training error should be reported separately.""",
    },
]

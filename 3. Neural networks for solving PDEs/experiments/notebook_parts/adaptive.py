"""Notebook chapter: a controlled experiment in residual-adaptive sampling."""
from textwrap import dedent

CELLS = []

def md(source):
    CELLS.append(dict(cell_type="markdown", source=dedent(source).strip()))

def code(source):
    CELLS.append(dict(cell_type="code", source=dedent(source).strip()))

md(r'''
## 11 · Choosing collocation points from the residual

A narrow solution feature can be missed by a small, fixed collocation set. We manufacture
the smooth solution
$$u(x)=4x(1-x)\exp\!\left[-\left(\frac{x-0.63}{0.025}\right)^2\right],
\qquad -u''=f,\qquad u(0)=u(1)=0.$$
The forcing $f$ is differentiated analytically from this expression. **No solution values
enter training.** We compare fixed uniform points, uniformly refreshed points, and
residual-adaptive points. The refreshed-uniform control separates the benefit of moving
points from the benefit of using residual information.

To isolate sampling from difficult hidden-layer optimization, use a **fixed sine hidden
layer with trainable output weights**:
$$u_\theta(x)=s\sum_{k=1}^{64}\theta_k\frac{\sin(k\pi x)}{(k\pi)^2}.$$
This is also a spectral trial space. We keep the sine features fixed and train
only their output weights. The inverse-eigenvalue scaling makes the
normalized residual linear in the sine features. All three runs share initial weights,
initial points, point count, Adam updates, and learning-rate schedule.
''')

code('''
ADAPTIVE_SETTINGS = dict(seed=71, points_per_step=40, candidate_count=1000,
    hidden_features=64, stages=24, steps_per_stage=100 if RUN_MODE == "quick" else 200,
    learning_rate=0.015, stage_lr_decay=0.75, uniform_mixture=0.35,
    center=0.63, width=0.025, residual_scale=1000.0)
ad_cfg = ADAPTIVE_SETTINGS
ad_dtype = torch.float64
ad_freq = torch.arange(1, ad_cfg["hidden_features"] + 1, dtype=ad_dtype)[None, :] * np.pi
ad_candidates = torch.tensor((np.arange(ad_cfg["candidate_count"]) + 0.5)[:, None]
                             / ad_cfg["candidate_count"], dtype=ad_dtype)
ad_x = np.linspace(0.0, 1.0, 1601)  # independent validation grid
ad_xt = torch.tensor(ad_x[:, None], dtype=ad_dtype)

def ad_reference(ad_input):
    ad_z = (ad_input - ad_cfg["center"]) / ad_cfg["width"]
    return 4 * ad_input * (1 - ad_input) * torch.exp(-ad_z.square())

def ad_forcing(ad_input):
    ad_w = ad_cfg["width"]
    ad_z = (ad_input - ad_cfg["center"]) / ad_w
    return -4 * torch.exp(-ad_z.square()) * (
        -2 - 4 * (1 - 2 * ad_input) * ad_z / ad_w
        + ad_input * (1 - ad_input) * (4 * ad_z.square() - 2) / ad_w**2)

class ad_SineNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_weights = torch.nn.Parameter(
            0.001 * torch.randn(ad_cfg["hidden_features"], 1, dtype=ad_dtype))

    def forward(self, ad_input):
        return ad_cfg["residual_scale"] * (
            torch.sin(ad_input * ad_freq) / ad_freq.square()) @ self.output_weights

# Cached fixed hidden features reduce repeated work; no reference state labels are used.
ad_phi = torch.sin(ad_candidates * ad_freq)
ad_f = ad_forcing(ad_candidates) / ad_cfg["residual_scale"]
ad_val_phi = torch.sin(ad_xt * ad_freq)
ad_val_f = ad_forcing(ad_xt)
ad_exact = ad_reference(ad_xt).numpy().ravel()
ad_initial_indices = (np.arange(ad_cfg["points_per_step"]) * 25 + 12).astype(int)
assert ad_cfg["candidate_count"] == 25 * ad_cfg["points_per_step"]
torch.manual_seed(ad_cfg["seed"])
ad_initial_model = ad_SineNetwork()
ad_initial_weights = ad_initial_model.output_weights.detach().clone()

# Verify the manufactured forcing once, independently by automatic differentiation.
ad_check_x = torch.linspace(0.51, 0.76, 23, dtype=ad_dtype)[:, None].requires_grad_()
ad_check_u = ad_reference(ad_check_x)
ad_check_du = torch.autograd.grad(ad_check_u.sum(), ad_check_x, create_graph=True)[0]
ad_check_d2u = torch.autograd.grad(ad_check_du.sum(), ad_check_x)[0]
assert torch.allclose(-ad_check_d2u, ad_forcing(ad_check_x), rtol=1e-11, atol=1e-9)
print(f"Each run: {ad_cfg['points_per_step']} points/update, "
      f"{ad_cfg['stages'] * ad_cfg['steps_per_stage']} Adam updates, identical initialization.")
''')

md(r'''
### Importance weights preserve the sampling objective

On a fixed uniform candidate grid of $M$ points, the target is
$\widehat{\mathcal J}_M(\theta)=M^{-1}\sum_i \widetilde r_\theta(x_i)^2$,
where $\widetilde r=r/s$. Before each new stage, form the frozen proposal
$$p_i=\frac{\alpha}{M}+(1-\alpha)
\frac{\widetilde r_{\rm old}(x_i)^2}{\sum_j\widetilde r_{\rm old}(x_j)^2},
\qquad\alpha=0.35.$$
Draw $N$ indices independently **with replacement**, then train for one stage on
$$\widehat{\mathcal J}_{\rm batch}(\theta)=\frac1N\sum_{b=1}^N
\frac{\widetilde r_\theta(x_{i_b})^2}{M p_{i_b}}.$$
For any fixed trial state, averaging over these draws recovers the uniform candidate
objective exactly. The proposal and importance weights stay frozen during the stage.
This expectation identity does **not** guarantee generalization after fitting a drawn
batch. The candidate objective itself is a numerical approximation to the continuum loss.
The mixture keeps every candidate selectable and bounds each weight by $1/\alpha$.

Adaptive sampling also scores all $M$ candidates between stages. We count those extra
residual evaluations and time the scans; equal training-point budgets are not equal total costs.
''')

code('''
def ad_run(ad_method):
    ad_rng = np.random.default_rng(ad_cfg["seed"])
    ad_model = ad_SineNetwork()
    with torch.no_grad():
        ad_model.output_weights.copy_(ad_initial_weights)
    ad_optimizer = torch.optim.Adam(ad_model.parameters(), lr=ad_cfg["learning_rate"])
    ad_indices = ad_initial_indices.copy()
    ad_weights = torch.ones((ad_cfg["points_per_step"], 1), dtype=ad_dtype)
    ad_hist = dict(relative_l2=[], residual_rms=[], sampled_loss=[], points=[], residual_map=[])
    ad_search_seconds, ad_scored_points = 0.0, 0
    ad_start = time.perf_counter()
    for ad_stage in range(ad_cfg["stages"]):
        for ad_group in ad_optimizer.param_groups:
            ad_group["lr"] = ad_cfg["learning_rate"] * ad_cfg["stage_lr_decay"]**ad_stage
        if ad_stage > 0 and ad_method != "Fixed uniform":
            if ad_method == "Residual adaptive":
                ad_search_start = time.perf_counter()
                with torch.no_grad():
                    ad_scores = (ad_phi @ ad_model.output_weights - ad_f).square().numpy().ravel()
                ad_prob = np.full(ad_cfg["candidate_count"], 1 / ad_cfg["candidate_count"])
                if ad_scores.sum() > 0:
                    ad_prob = (ad_cfg["uniform_mixture"] * ad_prob
                               + (1 - ad_cfg["uniform_mixture"]) * ad_scores / ad_scores.sum())
                ad_search_seconds += time.perf_counter() - ad_search_start
                ad_scored_points += ad_cfg["candidate_count"]
            else:
                ad_prob = np.full(ad_cfg["candidate_count"], 1 / ad_cfg["candidate_count"])
            ad_indices = ad_rng.choice(ad_cfg["candidate_count"], ad_cfg["points_per_step"],
                                       replace=True, p=ad_prob)
            ad_weights = torch.tensor((1 / (ad_cfg["candidate_count"] * ad_prob[ad_indices]))[:, None],
                                      dtype=ad_dtype)
        ad_batch_phi, ad_batch_f = ad_phi[ad_indices], ad_f[ad_indices]
        for ad_step in range(ad_cfg["steps_per_stage"]):
            ad_optimizer.zero_grad(set_to_none=True)
            ad_batch_r = ad_batch_phi @ ad_model.output_weights - ad_batch_f
            ad_loss = (ad_weights * ad_batch_r.square()).mean()
            ad_loss.backward()
            ad_optimizer.step()
        with torch.no_grad():
            ad_prediction = ad_model(ad_xt).numpy().ravel()
            ad_residual = (ad_cfg["residual_scale"] * ad_val_phi @ ad_model.output_weights
                           - ad_val_f).numpy().ravel()
            ad_err = np.sqrt(np.trapezoid((ad_prediction - ad_exact)**2, ad_x)
                             / np.trapezoid(ad_exact**2, ad_x))
            ad_rms = np.sqrt(np.trapezoid(ad_residual**2, ad_x))
            ad_final_batch = ad_batch_phi @ ad_model.output_weights - ad_batch_f
            ad_hist["sampled_loss"].append(float((ad_weights * ad_final_batch.square()).mean()))
        ad_hist["relative_l2"].append(float(ad_err))
        ad_hist["residual_rms"].append(float(ad_rms))
        ad_hist["points"].append(ad_candidates[ad_indices].numpy().ravel())
        ad_hist["residual_map"].append(ad_residual[::8] / ad_cfg["residual_scale"])
    ad_seconds = time.perf_counter() - ad_start
    return dict(prediction=ad_prediction, residual=ad_residual, relative_l2=float(ad_err),
        residual_rms=float(ad_rms), history=ad_hist, seconds=ad_seconds,
        candidate_scan_seconds=ad_search_seconds, extra_candidate_evaluations=ad_scored_points,
        training_point_evaluations=ad_cfg["points_per_step"] * ad_cfg["stages"] * ad_cfg["steps_per_stage"])

ad_results = {}
for ad_label in ["Fixed uniform", "Uniform refresh", "Residual adaptive"]:
    ad_result = ad_run(ad_label)
    ad_results[ad_label] = ad_result
    metrics.append(dict(experiment=f"Adaptive sampling / {ad_label}",
        relative_l2=ad_result["relative_l2"], seconds=ad_result["seconds"],
        residual_rms=ad_result["residual_rms"],
        extra_candidate_evaluations=ad_result["extra_candidate_evaluations"]))
    print(f"{ad_label:18s} | relative L2 {ad_result['relative_l2']:.2%} | "
          f"physical residual RMS {ad_result['residual_rms']:.3f} | {ad_result['seconds']:.2f} s")
print("Per run: %d training-point evaluations." % ad_result["training_point_evaluations"])
print("Adaptive extras: %d candidate residuals; %.4f s of scoring + proposal construction." % (
    ad_results["Residual adaptive"]["extra_candidate_evaluations"],
    ad_results["Residual adaptive"]["candidate_scan_seconds"]))
print("Times include identical validation schedules. Candidates and fixed features were cached once.")
''')

code('''
ad_colors = {"Fixed uniform": palette["orange"], "Uniform refresh": palette["blue"],
             "Residual adaptive": palette["teal"]}
ad_steps = (np.arange(ad_cfg["stages"]) + 1) * ad_cfg["steps_per_stage"]
ad_fig, ad_axes = plt.subplots(2, 2, figsize=(13.5, 8.0), layout="constrained")
ad_axes[0, 0].plot(ad_x, ad_exact, color=palette["ink"], lw=4, alpha=0.3, label="Reference")
for ad_label, ad_result in ad_results.items():
    ad_color = ad_colors[ad_label]
    ad_axes[0, 0].plot(ad_x, ad_result["prediction"], color=ad_color, label=ad_label)
    ad_axes[0, 1].semilogy(ad_x, np.maximum(np.abs(ad_result["prediction"] - ad_exact), 1e-8),
                         color=ad_color, label=ad_label)
    ad_axes[1, 0].semilogy(ad_steps, ad_result["history"]["relative_l2"], "o-",
                         color=ad_color, ms=3, label=ad_label)
    ad_axes[1, 1].semilogy(ad_steps, np.asarray(ad_result["history"]["residual_rms"]) /
                         ad_cfg["residual_scale"], "o-", color=ad_color, ms=3, label=ad_label)
ad_axes[0, 0].set(title="Recovering a narrow solution feature", xlabel="x", ylabel="u(x)")
ad_axes[0, 1].set(title="Error on an independent grid", xlabel="x", ylabel="Absolute state error")
ad_axes[1, 0].set(title="Point placement changes convergence", xlabel="Adam updates", ylabel="Relative L2 error")
ad_axes[1, 1].set(title="Check between the training points", xlabel="Adam updates", ylabel="Normalized residual RMS")
for ad_ax in ad_axes.ravel():
    ad_ax.grid(alpha=0.35)
    ad_ax.legend(fontsize=8)
ad_fig.suptitle("Same network and update budget · three ways to choose points", fontsize=17, weight="bold")
save_figure(ad_fig, "11_adaptive_accuracy")

ad_fig, ad_axes = plt.subplots(2, 2, figsize=(13.5, 7.5), layout="constrained")
ad_map_x = ad_x[::8]
ad_maps = {ad_label: np.log10(np.maximum(np.abs(np.asarray(ad_result["history"]["residual_map"])), 1e-5))
           for ad_label, ad_result in ad_results.items()}
ad_vmax = max(float(ad_map.max()) for ad_map in ad_maps.values())
for ad_ax, ad_label in zip(ad_axes[0], ["Fixed uniform", "Residual adaptive"]):
    ad_im = ad_ax.pcolormesh(ad_map_x, np.arange(1, ad_cfg["stages"] + 1), ad_maps[ad_label],
                            shading="nearest", cmap="magma", vmin=-5, vmax=ad_vmax)
    ad_ax.set(title=f"{ad_label} · residual map", xlabel="x", ylabel="Stage")
ad_fig.colorbar(ad_im, ax=list(ad_axes[0]), label="log10 |normalized residual|", shrink=0.85)
for ad_ax, ad_label in zip(ad_axes[1], ["Uniform refresh", "Residual adaptive"]):
    for ad_stage, ad_points in enumerate(ad_results[ad_label]["history"]["points"], start=1):
        ad_ax.scatter(ad_points, np.full_like(ad_points, ad_stage), s=12, alpha=0.65,
                       color=ad_colors[ad_label], linewidths=0)
    ad_ax.axvspan(ad_cfg["center"] - 2 * ad_cfg["width"],
                  ad_cfg["center"] + 2 * ad_cfg["width"], color=palette["orange"], alpha=0.10)
    ad_ax.set(title=f"{ad_label} · sampled locations", xlabel="x", ylabel="Stage", xlim=(0, 1))
ad_fig.suptitle("The residual landscape and the points that inspect it", fontsize=17, weight="bold")
save_figure(ad_fig, "11_adaptive_sampling_maps")
''')

md(r'''
Forty fixed points undersample this 64-feature trial space;
a tiny loss on those points can coexist with a large residual elsewhere. Uniform refresh
is therefore an essential baseline. The adaptive run can focus effort where the current
model is wrong while the importance weights preserve its *expected* candidate objective.
Inspect the stages where progress stalls or reverses, and repeat with other seeds
to see how stable the comparison is.

The extra candidate scans use 23,000 residual evaluations in quick mode, on top of 96,000
training-point evaluations. Cached sine features make those scans inexpensive here;
second derivatives through a fully trainable network can make searching much more costly.
The timing includes the same validation schedule for every method.

**Try:** (1) increase the fixed-point count and adjust the candidate-grid ratio; (2) replace
residual-adaptive probabilities by uniform probabilities and check the importance weights
become one; (3) repeat with several seeds before drawing a method comparison; (4) reduce
the uniform mixture and inspect both maximum weight and training stability; (5) increase
the number of sine features and ask whether sampling or approximation now limits accuracy.
''')

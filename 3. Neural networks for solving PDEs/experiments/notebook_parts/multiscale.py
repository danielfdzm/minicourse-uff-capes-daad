"""Chapter 10: a finite-budget multiscale Poisson/Fourier-feature comparison."""
from textwrap import dedent

CELLS = []
def md(s): CELLS.append(dict(cell_type="markdown", source=dedent(s).strip()))
def code(s): CELLS.append(dict(cell_type="code", source=dedent(s).strip()))

md(r'''
## 10 · Learning two spatial scales with Fourier features

Solve a one-dimensional Poisson problem with a slow wave and a small, faster ripple:
$$-u''=f,\quad u(0)=u(1)=0,\qquad
f(x)=\pi^2\sin(\pi x)+a(k\pi)^2\sin(k\pi x),\quad k=8,\ a=0.15.$$
The reference is $u(x)=\sin(\pi x)+a\sin(k\pi x)$, used **only for validation**.
Both neural trials use $u_\theta(x)=x(1-x)N_\theta(\Phi(x))$, so their boundary values
are exact. Compare plain coordinates $\Phi(x)=2x-1$ with
$$\Phi(x)=\bigl[2x-1,\sin(2\pi b x),\cos(2\pi b x)\bigr]_{b\in\{1,2,3\}}.$$
The Fourier features supply a small general frequency bank; the target ripple's
frequency $b=k/2=4$ is **not** explicitly included. The network must still learn its
coefficients and the interactions needed to fit the PDE.

We use the same interior points, seed, hidden widths, optimizer settings, and iteration
limits. The larger Fourier input has more parameters and costs more per evaluation:
compare the errors with this extra cost in mind. Differentiating a fast ripple twice
amplifies its contribution to the residual by $k^2$.
''')

code('''
ms_seed, ms_k, ms_amplitude = 314, 8, 0.15
ms_frequencies = [1., 2., 3.]
ms_point_count, ms_width, ms_depth = 128, 24, 2
ms_adam_steps = 450 if RUN_MODE == "quick" else 1200
ms_lbfgs_steps = 100 if RUN_MODE == "quick" else 260
ms_train_coordinates = (torch.arange(ms_point_count).reshape(-1, 1) + .5) / ms_point_count
ms_test_x = np.linspace(0., 1., 1001)
ms_test_tensor = torch.tensor(ms_test_x[:, None])
ms_modes = np.arange(1, 13)
ms_reference = np.sin(np.pi*ms_test_x) + ms_amplitude*np.sin(ms_k*np.pi*ms_test_x)
ms_reference_modes = np.zeros(len(ms_modes))
ms_reference_modes[0], ms_reference_modes[ms_k-1] = 1., ms_amplitude
ms_basis = np.sin(np.pi * ms_modes[:, None] * ms_test_x[None, :])
ms_forcing_scale = np.sqrt((np.pi**4 + (ms_amplitude*(ms_k*np.pi)**2)**2)/2)

def ms_forcing(x):
    return np.pi**2*torch.sin(np.pi*x) + ms_amplitude*(ms_k*np.pi)**2*torch.sin(ms_k*np.pi*x)

class ms_Trial(torch.nn.Module):
    def __init__(self, fourier=False):
        super().__init__()
        self.fourier = fourier
        self.register_buffer("frequencies", torch.tensor(ms_frequencies).reshape(1, -1))
        self.net = mlp(1 + 2*len(ms_frequencies) if fourier else 1,
                       width=ms_width, depth=ms_depth)
    def forward(self, x):
        features = 2*x-1
        if self.fourier:
            phases = 2*np.pi*x*self.frequencies
            features = torch.cat((features, torch.sin(phases), torch.cos(phases)), dim=1)
        return x*(1-x)*self.net(features)

def ms_residual(model, coordinates):
    x = coordinates.detach().clone().requires_grad_(True)
    return -grad_scalar(grad_scalar(model(x), x), x) - ms_forcing(x)

MULTISCALE_SETTINGS = dict(seed=ms_seed, k=ms_k, amplitude=ms_amplitude,
    feature_frequencies=ms_frequencies, points=ms_point_count,
    sampling="fixed interior midpoints", width=ms_width, hidden_layers=ms_depth,
    adam_steps=ms_adam_steps, adam_learning_rate=0.002, lbfgs_max_iter=ms_lbfgs_steps,
    residual_normalization=float(ms_forcing_scale), exact_boundary_factor="x*(1-x)",
    device="cpu", dtype="float64", threads=1)
assert torch.get_default_dtype() == torch.float64 and torch.get_num_threads() == 1
# Check the manufactured forcing independently through automatic differentiation.
ms_check_x = torch.linspace(.013, .987, 53).reshape(-1, 1).requires_grad_(True)
ms_check_u = torch.sin(np.pi*ms_check_x) + ms_amplitude*torch.sin(ms_k*np.pi*ms_check_x)
ms_source_error = (grad_scalar(grad_scalar(ms_check_u, ms_check_x), ms_check_x)
                   + ms_forcing(ms_check_x)).detach().abs().max().item()
assert ms_source_error < 1e-10
print(f"Manufactured-source check: {ms_source_error:.2e}")
''')

code('''
ms_results = {}
for ms_name, ms_fourier in [("Plain tanh", False), ("Fourier features", True)]:
    torch.manual_seed(ms_seed)
    ms_model = ms_Trial(fourier=ms_fourier)
    ms_modal_snapshots = []
    def ms_diagnostic():
        with torch.no_grad():
            prediction = ms_model(ms_test_tensor).numpy().ravel()
        ms_modal_snapshots.append(2*np.trapezoid(ms_basis*prediction[None, :], ms_test_x, axis=1))
        return float(np.linalg.norm(prediction-ms_reference)/np.linalg.norm(ms_reference))
    ms_initial_error = ms_diagnostic()
    def ms_objective():
        return (ms_residual(ms_model, ms_train_coordinates)/ms_forcing_scale).square().mean()
    ms_history = train_model(ms_model, ms_objective,
        adam_steps=ms_adam_steps, lbfgs_steps=ms_lbfgs_steps, diagnostic=ms_diagnostic)
    with torch.no_grad():
        ms_prediction = ms_model(ms_test_tensor).numpy().ravel()
        ms_boundary_error = ms_model(torch.tensor([[0.], [1.]])).abs().max().item()
    ms_test_residual = ms_residual(ms_model, ms_test_tensor).detach().numpy().ravel()
    ms_relative_error = float(np.sqrt(np.trapezoid((ms_prediction-ms_reference)**2, ms_test_x)
                                      / np.trapezoid(ms_reference**2, ms_test_x)))
    ms_residual_rms = float(np.sqrt(np.trapezoid(ms_test_residual**2, ms_test_x)))
    ms_parameter_count = sum(parameter.numel() for parameter in ms_model.parameters())
    ms_results[ms_name] = dict(model=ms_model, prediction=ms_prediction,
        residual=ms_test_residual, relative_l2=ms_relative_error, residual_rms=ms_residual_rms,
        history=ms_history, modal_snapshots=np.asarray(ms_modal_snapshots),
        snapshot_evaluations=np.array([0] + ms_history["error_eval"]),
        parameters=ms_parameter_count, boundary_max_abs=ms_boundary_error)
    assert np.isfinite(ms_prediction).all() and ms_boundary_error == 0.
    print(f"{ms_name:17s} | parameters {ms_parameter_count:4d} | "
          f"relative L2 {ms_relative_error:.3e} | test residual RMS {ms_residual_rms:.3e} | "
          f"{len(ms_history['loss'])} objective evaluations | {ms_history['seconds']:.2f} s")
MULTISCALE_SETTINGS["parameter_counts"] = {name: result["parameters"] for name, result in ms_results.items()}
''')

md(r'''
### Read the loss and the field together

The training loss is the **mean sampled squared residual**, normalized by the exact
source RMS squared. Residual curves below are evaluated on a separate dense grid and
shown in the PDE's original units. The reported $L^2$ errors and modal coefficients
use dense-grid trapezoidal integration for validation.

The sine coefficients $c_j=2\int_0^1u_\theta(x)\sin(j\pi x)\,dx$ expose whether the
network recovered both the large-scale component $(c_1=1)$ and the ripple $(c_8=0.15)$.
A slow component can remain wrong even after progress on the strongly weighted fast
residual. Change the feature bank, training budget, or seed to see whether the same
pattern appears.
''')

code(r'''
ms_colors = {"Plain tanh": palette["orange"], "Fourier features": palette["teal"]}
ms_fig, ms_axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
ms_axes[0, 0].plot(ms_test_x, ms_reference, color=palette["navy"], lw=2.5, label="Reference")
for ms_name, ms_result in ms_results.items():
    ms_color = ms_colors[ms_name]
    ms_axes[0, 0].plot(ms_test_x, ms_result["prediction"], color=ms_color, ls="--", label=ms_name)
    ms_axes[0, 1].plot(ms_test_x, ms_result["residual"], color=ms_color, alpha=.9, label=ms_name)
    ms_axes[1, 0].semilogy(ms_result["history"]["evaluation"], ms_result["history"]["loss"],
                          color=ms_color, label=ms_name)
ms_axes[0, 0].set(title="One state, two spatial scales", xlabel="x", ylabel="u(x)")
ms_axes[0, 1].set(title="Residual on unseen locations", xlabel="x", ylabel=r"$-u_\theta''-f$")
ms_axes[0, 1].axhline(0., color=palette["muted"], lw=.8, zorder=0)
ms_axes[1, 0].set(title="Same iteration limits; different evaluation counts",
                 xlabel="Objective evaluations (including line search)", ylabel="Normalized sampled loss")
ms_bar_width = .24
ms_axes[1, 1].bar(ms_modes-ms_bar_width, ms_reference_modes, width=ms_bar_width,
                  color=palette["navy"], label="Reference", alpha=.9)
for ms_offset, (ms_name, ms_result) in zip([0., ms_bar_width], ms_results.items()):
    ms_axes[1, 1].bar(ms_modes+ms_offset, ms_result["modal_snapshots"][-1], width=ms_bar_width,
                      color=ms_colors[ms_name], label=ms_name)
ms_axes[1, 1].set(title="Which Fourier modes were recovered?", xlabel="Sine mode j",
                 ylabel="Signed coefficient cⱼ", xticks=ms_modes)
for ms_ax in ms_axes.ravel():
    ms_ax.grid(alpha=.4); ms_ax.set_axisbelow(True); ms_ax.legend(fontsize=9)
ms_fig.suptitle("Multiscale Poisson · representation changes the optimization problem",
                fontsize=18, weight="bold")
save_figure(ms_fig, "10_multiscale_comparison")
''')

code('''
# Rotate the learning history: height is a recovered sine coefficient, not a PDE time axis.
ms_surfaces = []
for ms_index, (ms_name, ms_result) in enumerate(ms_results.items()):
    ms_surfaces.append(go.Surface(x=ms_modes, y=ms_result["snapshot_evaluations"],
        z=ms_result["modal_snapshots"], visible=(ms_index == 0), colorscale="Viridis",
        colorbar=dict(title="Coefficient cⱼ"), name=ms_name,
        hovertemplate="Mode j=%{x}<br>Objective evaluations=%{y}<br>Coefficient=%{z:.4f}<extra></extra>"))
ms_learning_fig = go.Figure(ms_surfaces)
ms_learning_fig.update_layout(title="Mode recovery during training · Plain tanh",
    scene=dict(xaxis_title="Sine mode j", yaxis_title="Loss evaluations",
               zaxis_title="Coefficient cⱼ", aspectratio=dict(x=1, y=1, z=.7),
               camera=dict(eye=dict(x=1.65, y=1.65, z=1.4))),
    updatemenus=[dict(type="buttons", direction="left", buttons=[
        dict(label=ms_name, method="update", args=[
            {"visible": [ms_index == j for j in range(2)]},
            {"title.text": f"Mode recovery during training · {ms_name}"}])
        for ms_index, ms_name in enumerate(ms_results)])])
show_interactive(ms_learning_fig, "10_multiscale_learning")
''')

code('''
for ms_name, ms_result in ms_results.items():
    metrics.append(dict(experiment=f"Multiscale: {ms_name}",
        relative_l2=ms_result["relative_l2"], residual_rms=ms_result["residual_rms"],
        parameters=ms_result["parameters"], seconds=ms_result["history"]["seconds"],
        objective_evaluations=len(ms_result["history"]["loss"]),
        boundary_max_abs=ms_result["boundary_max_abs"],
        mode_1=float(ms_result["modal_snapshots"][-1, 0]),
        mode_8=float(ms_result["modal_snapshots"][-1, 7])))
print("Recovered [slow mode, fast mode]; reference is [1.0, 0.15]:")
for ms_name, ms_result in ms_results.items():
    print(f"  {ms_name:17s}: {ms_result['modal_snapshots'][-1, [0, 7]]}")
print(json.dumps(MULTISCALE_SETTINGS, indent=2))
''')

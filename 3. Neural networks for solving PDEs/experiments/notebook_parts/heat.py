"""Time-dependent heat PINN cells for the standalone notebook."""
from textwrap import dedent

CELLS = []
def md(source):
    CELLS.append(dict(cell_type="markdown", source=dedent(source).strip()))
def code(source):
    CELLS.append(dict(cell_type="code", source=dedent(source).strip()))

md(r'''
## 12 · A time-dependent PINN: diffusion erases fine detail first

Train one neural field over **space and time**, rather than advancing a numerical state
through successive time steps. On $(x,t)\in(0,1)\times(0,1)$, solve
$$u_t=\nu u_{xx},\qquad \nu=0.08,\qquad u(0,t)=u(1,t)=0,$$
with the prescribed initial state
$$u_0(x)=\sin(\pi x)+0.25\sin(3\pi x).$$
The trial $u_\theta(x,t)=u_0(x)+t\,x(1-x)N_\theta(x,t)$ satisfies the initial and
boundary conditions exactly. Training uses **only the PDE residual and the given initial
condition**; there are no future solution labels. Scrambled Sobol points sample the
space–time rectangle.

For independent validation, separation of variables gives
$$u(x,t)=e^{-\nu\pi^2t}\sin(\pi x)
+0.25e^{-9\nu\pi^2t}\sin(3\pi x).$$
Mode $m$ decays at rate $\nu(m\pi)^2$: the third mode decays nine times faster
than the first. We measure the amplitudes by projecting the learned field onto
$\sin(m\pi x)$ on a fresh grid. The 3D surface has axes **space, time, and state**;
the physical domain is one-dimensional.
''')

code('''
ht_seed, ht_nu = 121, 0.08
HEAT_SETTINGS = dict(seed=ht_seed, diffusivity=ht_nu, points=768,
                     width=32, depth=2, adam_steps=1200, lbfgs_steps=240,
                     final_time=1.0, trial="u0(x) + t*x*(1-x)*N(x,t)")
if RUN_MODE == "thorough":
    HEAT_SETTINGS["adam_steps"] *= 2
    HEAT_SETTINGS["lbfgs_steps"] *= 2
torch.manual_seed(ht_seed)

def ht_initial(x):
    return torch.sin(np.pi*x) + 0.25*torch.sin(3*np.pi*x)

class HeatPINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(2, width=HEAT_SETTINGS["width"], depth=HEAT_SETTINGS["depth"])
    def forward(self, xt):
        x, t = xt[:, :1], xt[:, 1:2]
        return ht_initial(x) + t*x*(1-x)*self.net(2*xt-1)

def ht_residual(model, xt):
    xt = xt.detach().requires_grad_()
    value = model(xt)
    derivative = grad_scalar(value, xt)
    second_x = grad_scalar(derivative[:, :1], xt)[:, :1]
    return derivative[:, 1:2] - ht_nu*second_x

ht_model = HeatPINN()
ht_training = torch.quasirandom.SobolEngine(2, scramble=True, seed=ht_seed).draw(
    HEAT_SETTINGS["points"]).double()
ht_history = train_model(ht_model, lambda: ht_residual(ht_model, ht_training).square().mean(),
                        HEAT_SETTINGS["adam_steps"], HEAT_SETTINGS["lbfgs_steps"])
print(f"Trained a space–time field in {ht_history['seconds']:.1f} seconds.")
''')

code('''
# Independent regular-grid validation; the reference was not used for optimization.
ht_x = np.linspace(0, 1, 201)
ht_t = np.linspace(0, 1, 101)
ht_X, ht_T = np.meshgrid(ht_x, ht_t)
ht_xt = torch.tensor(np.column_stack((ht_X.ravel(), ht_T.ravel())))
with torch.no_grad():
    ht_prediction = ht_model(ht_xt).numpy().reshape(ht_X.shape)
ht_exact = (np.exp(-ht_nu*np.pi**2*ht_T)*np.sin(np.pi*ht_X)
            + 0.25*np.exp(-9*ht_nu*np.pi**2*ht_T)*np.sin(3*np.pi*ht_X))
ht_error = ht_prediction-ht_exact
ht_integral = lambda z: np.trapezoid(np.trapezoid(z, ht_x, axis=1), ht_t)
ht_relative_l2 = float(np.sqrt(ht_integral(ht_error**2)/ht_integral(ht_exact**2)))
# Evaluate residuals in batches to keep higher-derivative memory modest.
ht_test_residual = np.concatenate([
    ht_residual(ht_model, block).detach().numpy().ravel()
    for block in ht_xt.split(2048)
]).reshape(ht_X.shape)
ht_residual_rms = float(np.sqrt(ht_integral(ht_test_residual**2)))
ht_amplitudes = {m: 2*np.trapezoid(ht_prediction*np.sin(m*np.pi*ht_X), ht_x, axis=1)
                 for m in (1, 3)}
ht_exact_amplitudes = {1: np.exp(-ht_nu*np.pi**2*ht_t),
                       3: 0.25*np.exp(-9*ht_nu*np.pi**2*ht_t)}
ht_initial_error = float(np.max(np.abs(ht_error[0])))
ht_boundary_error = float(np.max(np.abs(ht_prediction[:, [0, -1]])))
assert ht_initial_error < 1e-12 and ht_boundary_error < 1e-12
assert np.isfinite(ht_test_residual).all()
metrics.append(dict(experiment="Heat space-time PINN", relative_l2=ht_relative_l2,
                    residual_rms=ht_residual_rms, seconds=ht_history["seconds"]))
print(f"Independent relative space–time L² error: {ht_relative_l2:.3%}")
print(f"Independent PDE residual RMS: {ht_residual_rms:.3e}")
print(f"Initial / boundary errors: {ht_initial_error:.1e} / {ht_boundary_error:.1e}")

ht_fig = plt.figure(figsize=(16, 5.2), layout="constrained")
ht_ax0 = ht_fig.add_subplot(131, projection="3d")
ht_ax0.plot_surface(ht_X, ht_T, ht_prediction, cmap="viridis",
                    linewidth=0, antialiased=True, rcount=80, ccount=100)
ht_ax0.set(title="One neural field across space and time", xlabel="Space x",
            ylabel="Time t")
ht_ax0.text2D(0.04, 0.90, r"$u_\\theta(x,t)$", transform=ht_ax0.transAxes,
              color=palette["navy"], fontsize=12)
ht_ax0.view_init(elev=28, azim=-124)
ht_ax0.set_box_aspect((1, 1, 0.7))
ht_ax1 = ht_fig.add_subplot(132)
ht_im = ht_ax1.pcolormesh(ht_X, ht_T, np.abs(ht_error), cmap="magma", shading="auto")
ht_fig.colorbar(ht_im, ax=ht_ax1, label="Absolute error", shrink=0.8)
ht_ax1.set(title=f"Unseen-grid error · relative L² {ht_relative_l2:.2%}",
            xlabel="Space x", ylabel="Time t")
ht_ax2 = ht_fig.add_subplot(133)
for ht_m, ht_color in [(1, palette["teal"]), (3, palette["orange"])]:
    ht_ax2.plot(ht_t, ht_exact_amplitudes[ht_m], color=ht_color, lw=2.7,
                label=f"Mode {ht_m} · exact")
    ht_ax2.plot(ht_t[::8], ht_amplitudes[ht_m][::8], 'o', color=ht_color,
                markerfacecolor="white", ms=5, label=f"Mode {ht_m} · neural")
ht_ax2.set(title="High frequencies disappear faster", xlabel="Time t", ylabel="Sine-mode amplitude")
ht_ax2.legend(fontsize=9)
ht_ax2.grid(alpha=0.3)
ht_fig.suptitle("Heat flow smooths the initial oscillations", fontsize=19,
                weight="bold", color=palette["navy"])
save_figure(ht_fig, "12_heat_space_time")
''')

code('''
# A time slider compares profiles with a fixed vertical scale throughout.
ht_frame_indices = np.linspace(0, len(ht_t)-1, 41).astype(int)
ht_profile_fig = go.Figure(data=[
    go.Scatter(x=ht_x, y=ht_exact[0], mode="lines", name="Exact heat flow",
                line=dict(color=palette["teal"], width=4), fill="tozeroy",
                fillcolor="rgba(0,166,166,0.13)"),
    go.Scatter(x=ht_x[::4], y=ht_prediction[0, ::4], mode="markers", name="Neural field",
                marker=dict(color=palette["orange"], size=6,
                            line=dict(color="white", width=0.5))),
    go.Scatter(x=ht_x, y=ht_exact[0], mode="lines", name="Initial profile",
                line=dict(color=palette["muted"], dash="dot", width=1.6)),
], frames=[go.Frame(name=str(i), data=[go.Scatter(y=ht_exact[i]),
             go.Scatter(y=ht_prediction[i, ::4])], traces=[0, 1]) for i in ht_frame_indices])
ht_profile_fig.update_layout(
    title="Heat PINN · watch the fine-scale detail dissolve",
    xaxis=dict(title="Space x", range=[0, 1]),
    yaxis=dict(title="State u(x,t)", range=[-0.025, 1.05]),
    legend=dict(orientation="h", x=0, y=1.03),
    updatemenus=[dict(type="buttons", direction="left", x=0, y=1.18, buttons=[
        dict(label="▶ Diffuse", method="animate", args=[None,
             dict(frame=dict(duration=90, redraw=False), transition=dict(duration=0), fromcurrent=True)]),
        dict(label="Pause", method="animate", args=[[None], dict(mode="immediate",
             frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
    ])],
    sliders=[dict(currentvalue=dict(prefix="Time t = "), steps=[
        dict(label=f"{ht_t[i]:.2f}", method="animate", args=[[str(i)],
             dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])
        for i in ht_frame_indices])],
)
show_interactive(ht_profile_fig, "12_heat_profiles_interactive")
''')

md(r'''
The network is trained over the whole time interval, so we can evaluate its profiles
at any time after training. That convenience comes with a global optimization cost.
For a cost comparison, run a classical time integrator to the same accuracy. Here,
the initial and boundary conditions hold exactly; the independent residual and
solution checks measure the remaining approximation and training errors.

**Try:**

- Increase $\nu$ and rerun. The rapid initial transient becomes harder to resolve with a
  fixed set of uniformly distributed time samples. Does sampling more points near $t=0$ help?
- Replace the third mode by the fifth in **both** the initial condition and validation
  formula. Update its decay rate to $25\nu\pi^2$ and project onto the fifth mode.
- Double the final time, update the training and validation time coordinates consistently,
  and compare errors in the early and late parts of the interval.
- Compare with a finite-difference heat solver. Report wall time, state error, and how
  each method stores or reconstructs values between its sampled times.
''')

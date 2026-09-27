"""A parameter-conditioned neural solver trained on a family of PDEs."""
from textwrap import dedent
CELLS = []
def md(s): CELLS.append(dict(cell_type='markdown', source=dedent(s).strip()))
def code(s): CELLS.append(dict(cell_type='code', source=dedent(s).strip()))

md(r'''
## 13 · Train one network, solve a family of PDEs

A network can accept a PDE parameter as an input. Consider the reaction–diffusion family
$$-u_{xx}+\mu u=\pi^2\sin(\pi x),\qquad u(0;\mu)=u(1;\mu)=0,
\quad x\in[0,1],\quad\mu\in[0,30].$$
Its exact state is
$$u(x;\mu)=\frac{\pi^2}{\pi^2+\mu}\sin(\pi x).$$
Train one $u_\theta(x,\mu)=x(1-x)N_\theta(x,\mu)$ using residual samples in the
joint $(x,\mu)$ domain. The derivative $u_{xx}$ is taken **only with respect to the
spatial coordinate**; $\mu$ is a parameter, not a second spatial dimension.
No solution labels enter training. After training, new parameter values require only
network evaluation. This **parameter-conditioned PINN** is trained for the stated forcing family.
Changing the forcing would require new training or additional inputs.

For this example, the analytic formula is fastest. The network lets us study how
one training run can serve many parameter queries. We compare its errors with a
tridiagonal finite-difference solver; assessing speed would also require counting
the training cost.
''')
code('''
PARAMETRIC_SETTINGS = dict(seed=1313, mu_max=30., collocation=768, width=32, depth=3,
                           adam_steps=1000, lbfgs_steps=220, fd_interior_nodes=63)
if RUN_MODE == "thorough":
    PARAMETRIC_SETTINGS["adam_steps"] *= 2
    PARAMETRIC_SETTINGS["lbfgs_steps"] *= 2
pa_cfg = PARAMETRIC_SETTINGS
torch.manual_seed(pa_cfg["seed"])

class pa_FamilyPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(2, width=pa_cfg["width"], depth=pa_cfg["depth"])
    def forward(self, xm):
        x, mu = xm[:, :1], xm[:, 1:2]
        features = torch.cat([2*x-1, 2*mu/pa_cfg["mu_max"]-1], dim=1)
        return x*(1-x)*self.net(features)

def pa_residual(model, xm):
    xm = xm.detach().requires_grad_()
    u = model(xm)
    ux = grad_scalar(u, xm)[:, :1]
    uxx = grad_scalar(ux, xm)[:, :1]
    return (-uxx+xm[:, 1:2]*u-np.pi**2*torch.sin(np.pi*xm[:, :1]))/np.pi**2

pa_model = pa_FamilyPINN()
pa_train = torch.quasirandom.SobolEngine(2, scramble=True, seed=pa_cfg["seed"]).draw(
                        pa_cfg["collocation"], dtype=torch.float64)
pa_train[:, 1] *= pa_cfg["mu_max"]
pa_history = train_model(pa_model, lambda: pa_residual(pa_model, pa_train).square().mean(),
                         pa_cfg["adam_steps"], pa_cfg["lbfgs_steps"])
pa_x = np.linspace(0, 1, 201)
pa_mu = np.linspace(0, pa_cfg["mu_max"], 61)
pa_X, pa_M = np.meshgrid(pa_x, pa_mu)
pa_inputs = torch.tensor(np.c_[pa_X.ravel(), pa_M.ravel()])
with torch.no_grad():
    pa_prediction = pa_model(pa_inputs).numpy().reshape(pa_X.shape)
pa_exact = np.pi**2/(np.pi**2+pa_M)*np.sin(np.pi*pa_X)
pa_error = pa_prediction-pa_exact
pa_param_errors = np.sqrt(np.trapezoid(pa_error**2, pa_x, axis=1)
                          / np.trapezoid(pa_exact**2, pa_x, axis=1))
pa_relative_l2 = float(np.sqrt(
    np.trapezoid(np.trapezoid(pa_error**2,pa_x,axis=1),pa_mu) /
    np.trapezoid(np.trapezoid(pa_exact**2,pa_x,axis=1),pa_mu)))
pa_normalized_rms = float(torch.sqrt(pa_residual(pa_model,pa_inputs).square().mean()).detach())
assert np.max(np.abs(pa_prediction[:,[0,-1]])) == 0
assert np.isfinite(pa_error).all()
metrics.append(dict(experiment="Parameter-conditioned PINN", relative_l2=pa_relative_l2,
                     maximum_parameter_error=float(pa_param_errors.max()),
                     normalized_residual_rms=pa_normalized_rms,seconds=pa_history["seconds"]))
print(f"One trained model, {len(pa_mu)} independently evaluated parameter slices.")
print(f"Relative L2 over (x, mu): {pa_relative_l2:.3e}; worst slice: {pa_param_errors.max():.3e}")
print(f"Training: {pa_history['seconds']:.1f} s; parameters: {sum(p.numel() for p in pa_model.parameters())}")
''')
code('''
# Solve the same parameter slices with second-order finite differences.
pa_fd_n = pa_cfg["fd_interior_nodes"]
pa_fd_x = np.linspace(0,1,pa_fd_n+2)
pa_fd_h = 1/(pa_fd_n+1)
pa_fd_predictions = []
for mu in pa_mu:
    band = np.zeros((3,pa_fd_n))
    band[0,1:],band[1],band[2,:-1] = -1.,2.+mu*pa_fd_h**2,-1.
    values = np.r_[0.,solve_banded((1,1),band,pa_fd_h**2*np.pi**2*np.sin(np.pi*pa_fd_x[1:-1])),0.]
    pa_fd_predictions.append(np.interp(pa_x,pa_fd_x,values))
pa_fd_predictions = np.asarray(pa_fd_predictions)
pa_fd_errors = np.sqrt(np.trapezoid((pa_fd_predictions-pa_exact)**2,pa_x,axis=1)
                      / np.trapezoid(pa_exact**2,pa_x,axis=1))

fig = plt.figure(figsize=(14,8),layout="constrained")
ax=fig.add_subplot(221,projection="3d")
ax.plot_surface(pa_X,pa_M,pa_prediction,cmap=sea_cmap,linewidth=0,rcount=61,ccount=100)
style_3d(ax,"One neural surface contains a family of solutions")
ax.set(xlabel="Position x",ylabel="Reaction μ",zlabel="uθ(x; μ)")
ax=fig.add_subplot(222)
limit=np.max(np.abs(pa_error))
im=ax.pcolormesh(pa_x,pa_mu,pa_error,cmap="RdBu_r",vmin=-limit,vmax=limit,shading="auto")
fig.colorbar(im,ax=ax,label="Signed error",shrink=.9)
ax.set(title="Check every region of parameter space",xlabel="Position x",ylabel="Reaction μ")
ax=fig.add_subplot(223)
for idx,color in zip([5,25,55],[palette["teal"],palette["orange"],palette["pink"]]):
    ax.plot(pa_x,pa_prediction[idx],color=color,label=f"NN · μ={pa_mu[idx]:g}")
    ax.plot(pa_x[::10],pa_exact[idx,::10],"o",mfc="none",mec=color,ms=4)
ax.set(title="New parameter queries after training",xlabel="x",ylabel="u(x; μ)")
ax.text(.03,.05,"Hollow points: exact reference",transform=ax.transAxes,color=palette["muted"],fontsize=9)
ax.legend(fontsize=9);ax.grid()
ax=fig.add_subplot(224)
ax.semilogy(pa_mu,pa_param_errors,color=palette["teal"],label="One conditioned PINN")
ax.semilogy(pa_mu,pa_fd_errors,color=palette["blue"],ls="--",label="FD: 63 interior nodes + interpolation")
ax.set(title="Accuracy varies across the family",xlabel="Reaction μ",ylabel="Relative L2 error per slice")
ax.legend(fontsize=9);ax.grid()
save_figure(fig,"13_parametric_solver")
''')
code('''
# Rotate the learned family, or colour the same surface by its error.
pa_surface = go.Figure([
    go.Surface(x=pa_X,y=pa_M,z=pa_prediction,colorscale="Viridis",name="Learned family",
                colorbar=dict(title="uθ"),visible=True),
    go.Surface(x=pa_X,y=pa_M,z=pa_prediction,surfacecolor=np.abs(pa_error),
                customdata=pa_error,colorscale="Magma",colorbar=dict(title="|Error|"),
                name="Error colours",visible=False,
                hovertemplate="x=%{x:.3f}<br>μ=%{y:.2f}<br>uθ=%{z:.5f}<br>Error=%{customdata:.2e}<extra></extra>")])
pa_surface.update_layout(title="One network · a continuous family of PDE solutions",
    scene=dict(xaxis_title="x",yaxis_title="Reaction μ",zaxis_title="uθ(x; μ)",
               aspectratio=dict(x=1,y=1,z=.65),uirevision="parameter-family"),
    updatemenus=[dict(buttons=[
        dict(label="Learned family",method="update",args=[dict(visible=[True,False])]),
        dict(label="Error colours",method="update",args=[dict(visible=[False,True])])])])
show_interactive(pa_surface,"13_parametric_interactive")
''')
md(r'''
**Try:** widen the parameter interval while keeping the training budget fixed. Where
does the worst error occur? Evaluate beyond $\mu=30$ and label it as **extrapolation**.
Then change the forcing to include a second sine mode: the current model has not been
trained for that new family. How would you add forcing coefficients as extra inputs?
''')

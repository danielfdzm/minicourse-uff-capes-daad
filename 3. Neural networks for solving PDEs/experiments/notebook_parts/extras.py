"""Additional computed notebook experiments: high dimension and curved domains."""

CELLS = [
    {
        "cell_type": "markdown",
        "source": r"""## Extra experiment — dimension, sampling, and the price of a grid

A tensor grid with 16 nodes in each direction needs $16^d$ points. Neural PDE methods often replace this grid by randomly sampled points. What does that buy us **before** we train a network?

Take the manufactured smooth field
$$u_d(x)=d^{-1/2}\sum_{k=1}^d\sin(\pi x_k),\qquad x\in(0,1)^d.$$
It solves $-\Delta u_d=\pi^2u_d$ with its own prescribed boundary trace. Its Dirichlet energy is exactly
$$\mathcal E_d=\frac12\int_{(0,1)^d}|\nabla u_d|^2\,dx=\frac{\pi^2}{4}.$$
For independent uniform samples, the relative RMS integration error is $1/\sqrt{2dN}$. We estimate this error using 16 independent runs and compare it with the formula. This tests the **integration step** used in neural PDE solvers. The error decreases with $d$ because of this field's additive structure; more general fields need separate analysis. The second panel compares point counts only, with no attempt to match accuracy.""",
    },
    {
        "cell_type": "code",
        "source": r"""# Actual Monte Carlo energy estimates; no training or saved results are used.
hd_rng = np.random.default_rng(1701)
hd_dims = np.array([2, 10, 50, 100])
hd_counts = np.array([128, 512, 2048, 8192])
hd_repeats = 16
hd_energy = np.pi**2 / 4
hd_rmse = np.zeros((len(hd_dims), len(hd_counts)))
for hd_i, hd_d in enumerate(hd_dims):
    hd_errors = []
    for hd_repeat in range(hd_repeats):
        hd_x = hd_rng.random((hd_counts[-1], hd_d))
        hd_density = 0.5 * np.pi**2 * np.mean(np.cos(np.pi * hd_x)**2, axis=1)
        hd_estimates = np.cumsum(hd_density)[hd_counts - 1] / hd_counts
        hd_errors.append(hd_estimates / hd_energy - 1)
    hd_rmse[hd_i] = np.sqrt(np.mean(np.array(hd_errors)**2, axis=0))

hd_colors = [palette['teal'], palette['orange'], palette['pink'], palette['blue']]
hd_fig, hd_ax = plt.subplots(1, 2, figsize=(14, 5.1), constrained_layout=True)
for hd_i, (hd_d, hd_color) in enumerate(zip(hd_dims, hd_colors)):
    hd_ax[0].loglog(hd_counts, hd_rmse[hd_i], 'o-', color=hd_color,
                   lw=2.4, ms=6, label=f'd = {hd_d}')
    hd_ax[0].loglog(hd_counts, 1 / np.sqrt(2 * hd_d * hd_counts), '--',
                   color=hd_color, alpha=0.45, lw=1.3)
hd_ax[0].set(title='Energy integration: measured error vs. theory',
             xlabel='Random points N', ylabel='Relative RMS error (16 runs)')
hd_ax[0].legend(ncol=2, frameon=False)
hd_ax[0].text(0.04, 0.05, 'Dashed: exact RMS prediction',
              transform=hd_ax[0].transAxes, color=palette['muted'])
hd_ax[1].plot(hd_dims, hd_dims * np.log10(16), 'o-', color=palette['pink'],
              lw=2.8, ms=7, label='Tensor grid: 16 nodes per coordinate')
hd_ax[1].axhline(np.log10(hd_counts[-1]), color=palette['teal'], lw=2.4,
                 label='Monte Carlo: 8,192 points')
hd_ax[1].set(title='The point budget grows very differently',
             xlabel='Dimension d', ylabel=r'$\log_{10}$(number of points)')
hd_ax[1].legend(frameon=False, loc='upper left')
hd_fig.suptitle('High dimension: sampling can bypass a tensor grid',
                fontsize=19, fontweight='bold', color=palette['navy'])
save_figure(hd_fig, 'extra_high_dimension')
print(f'Exact energy = {hd_energy:.8f}')
for hd_d, hd_row in zip(hd_dims, hd_rmse):
    print(f'd = {hd_d:3d} | relative RMS error at N = 8192: {hd_row[-1]:.3%}')
""",
    },
    {
        "cell_type": "markdown",
        "source": r"""## Extra experiment — a neural Poisson solution around a hole

Next, solve a PDE on an annulus, a domain with two circular boundaries. Let $a=0.35$, $b=1$, $s=x^2+y^2$, and
$$\Omega=\{(x,y):a<\sqrt{s}<b\},\qquad
\phi(x,y)=(s-a^2)(b^2-s).$$
We manufacture the exact solution $u=\phi(1+0.35x-0.20y)$ and obtain
$$-\Delta u=f,\quad u|_{\partial\Omega}=0,\qquad
f=16s-4c+(24s-8c)(0.35x-0.20y),\quad c=a^2+b^2.$$
The neural trial is $u_\theta=\phi N_\theta(x,y)$: both circular boundary conditions hold exactly, so training only minimizes an interior strong residual. We sample $r=\sqrt{a^2+(b^2-a^2)U}$ and $\vartheta=2\pi V$ with uniform independent $U,V$. This gives **uniform area sampling**; a uniform radius would overweight the inner part of the annulus.

Here we can construct sample points and a boundary factor directly, so we can train without a volume mesh. More complicated geometries may make either step difficult. A comparison with FEM would also need to measure solution error and total cost.""",
    },
    {
        "cell_type": "code",
        "source": r"""# A fresh small network and a deterministic area-uniform training set.
import time
torch.manual_seed(1702)
geo_rng = np.random.default_rng(1702)
geo_a, geo_b = 0.35, 1.0
geo_dtype = torch.float64
geo_uv = geo_rng.random((512, 2))
geo_radius = np.sqrt(geo_a**2 + (geo_b**2 - geo_a**2) * geo_uv[:, 0])
geo_angle = 2 * np.pi * geo_uv[:, 1]
geo_xy_np = np.column_stack((geo_radius * np.cos(geo_angle),
                              geo_radius * np.sin(geo_angle)))
geo_xy = torch.tensor(geo_xy_np, dtype=geo_dtype, requires_grad=True)
geo_net = torch.nn.Sequential(
    torch.nn.Linear(2, 24), torch.nn.Tanh(),
    torch.nn.Linear(24, 24), torch.nn.Tanh(),
    torch.nn.Linear(24, 1),
).to(dtype=geo_dtype)

def geo_phi(xy):
    s = (xy**2).sum(dim=1, keepdim=True)
    return (s - geo_a**2) * (geo_b**2 - s)

def geo_trial(xy):
    return geo_phi(xy) * geo_net(xy)

def geo_exact(xy):
    return geo_phi(xy) * (1 + 0.35 * xy[:, :1] - 0.20 * xy[:, 1:2])

def geo_forcing(xy):
    s = (xy**2).sum(dim=1, keepdim=True)
    c = geo_a**2 + geo_b**2
    q = 0.35 * xy[:, :1] - 0.20 * xy[:, 1:2]
    return 16 * s - 4 * c + (24 * s - 8 * c) * q

def geo_laplacian(values, xy, create_graph=True):
    grad = torch.autograd.grad(values.sum(), xy, create_graph=True)[0]
    return sum(torch.autograd.grad(grad[:, j].sum(), xy,
                                   create_graph=create_graph, retain_graph=True)[0][:, j:j+1]
               for j in range(2))

geo_forcing_train = geo_forcing(geo_xy).detach()
# Check the manufactured source independently by automatic differentiation.
geo_source_check = (geo_laplacian(geo_exact(geo_xy), geo_xy) +
                    geo_forcing_train).detach().abs().max().item()
assert geo_source_check < 1e-10
print(f'Manufactured-source check: max |Δu + f| = {geo_source_check:.2e}')
""",
    },
    {
        "cell_type": "code",
        "source": r"""# Full-batch L-BFGS. The recorded values include line-search evaluations.
geo_started = time.perf_counter()
geo_history = []
geo_optimizer = torch.optim.LBFGS(geo_net.parameters(), lr=1.0, max_iter=130,
                                 max_eval=180, history_size=30,
                                 tolerance_grad=1e-10, tolerance_change=1e-13,
                                 line_search_fn='strong_wolfe')

def geo_closure():
    geo_optimizer.zero_grad(set_to_none=True)
    if geo_xy.grad is not None:
        geo_xy.grad = None
    residual = -geo_laplacian(geo_trial(geo_xy), geo_xy) - geo_forcing_train
    loss = (residual**2).mean()
    loss.backward()
    geo_history.append(float(loss.detach()))
    return loss

geo_optimizer.step(geo_closure)
geo_seconds = time.perf_counter() - geo_started

# Independent, area-uniform validation points; training samples are not reused.
geo_test_uv = geo_rng.random((4096, 2))
geo_test_r = np.sqrt(geo_a**2 + (geo_b**2 - geo_a**2) * geo_test_uv[:, 0])
geo_test_t = 2 * np.pi * geo_test_uv[:, 1]
geo_test_xy = torch.tensor(np.column_stack((geo_test_r * np.cos(geo_test_t),
                                            geo_test_r * np.sin(geo_test_t))),
                             dtype=geo_dtype, requires_grad=True)
geo_test_pred = geo_trial(geo_test_xy)
geo_test_true = geo_exact(geo_test_xy)
geo_relative_l2 = float((torch.linalg.vector_norm(geo_test_pred - geo_test_true) /
                         torch.linalg.vector_norm(geo_test_true)).detach())
geo_test_res = -geo_laplacian(geo_test_pred, geo_test_xy, create_graph=False) - geo_forcing(geo_test_xy)
geo_test_res_rms = float(torch.sqrt(torch.mean(geo_test_res**2)).detach())
print(f'Training time: {geo_seconds:.1f} s | closure evaluations: {len(geo_history)}')
print(f'Independent relative L² error: {geo_relative_l2:.3%}')
print(f'Independent residual RMS: {geo_test_res_rms:.3e}')
print('Both boundary circles satisfy uθ = 0 by construction.')
""",
    },
    {
        "cell_type": "code",
        "source": r"""# A curved surface, a spatial error map, and the actual collocation points.
geo_plot_r, geo_plot_t = np.meshgrid(np.linspace(geo_a, geo_b, 80),
                                    np.linspace(0, 2 * np.pi, 181))
geo_plot_x, geo_plot_y = geo_plot_r * np.cos(geo_plot_t), geo_plot_r * np.sin(geo_plot_t)
geo_plot_xy = torch.tensor(np.column_stack((geo_plot_x.ravel(), geo_plot_y.ravel())),
                             dtype=geo_dtype)
with torch.no_grad():
    geo_plot_u = geo_trial(geo_plot_xy).numpy().reshape(geo_plot_x.shape)
    geo_plot_exact = geo_exact(geo_plot_xy).numpy().reshape(geo_plot_x.shape)
geo_grid_line = np.linspace(-1.04, 1.04, 230)
geo_grid_x, geo_grid_y = np.meshgrid(geo_grid_line, geo_grid_line)
geo_grid_s = geo_grid_x**2 + geo_grid_y**2
geo_mask = (geo_grid_s < geo_a**2) | (geo_grid_s > geo_b**2)
geo_grid_xy = torch.tensor(np.column_stack((geo_grid_x.ravel(), geo_grid_y.ravel())),
                             dtype=geo_dtype)
with torch.no_grad():
    geo_grid_err = (geo_trial(geo_grid_xy) - geo_exact(geo_grid_xy)).abs().numpy().reshape(geo_grid_x.shape)
geo_grid_err = np.ma.array(geo_grid_err, mask=geo_mask)
geo_fig = plt.figure(figsize=(17, 5.5), constrained_layout=True)
geo_ax0 = geo_fig.add_subplot(1, 3, 1, projection='3d')
geo_surf = geo_ax0.plot_surface(geo_plot_x, geo_plot_y, geo_plot_u,
                               cmap='viridis', linewidth=0, antialiased=True,
                               rcount=110, ccount=80, alpha=0.97)
style_3d(geo_ax0, 'Neural solution on the annulus')
geo_ax0.set(xlabel='x', ylabel='y', zlabel=r'$u_\theta$')
geo_ax0.view_init(elev=31, azim=-58)
geo_ax1 = geo_fig.add_subplot(1, 3, 2)
geo_im = geo_ax1.contourf(geo_grid_x, geo_grid_y, geo_grid_err,
                         levels=35, cmap='magma')
geo_fig.colorbar(geo_im, ax=geo_ax1, shrink=0.72, pad=0.035, label='Absolute error')
geo_ax1.set(title=f'Error on unseen locations · L² = {geo_relative_l2:.2%}',
             xlabel='x', ylabel='y', aspect='equal')
geo_ax2 = geo_fig.add_subplot(1, 3, 3)
geo_ax2.scatter(geo_xy_np[:, 0], geo_xy_np[:, 1], c=geo_radius,
                cmap='viridis', s=12, alpha=0.72, edgecolors='none')
geo_ax2.set(title='512 area-uniform training points', xlabel='x', ylabel='y', aspect='equal')
geo_circle_t = np.linspace(0, 2 * np.pi, 300)
for geo_ax in (geo_ax1, geo_ax2):
    for geo_r_boundary in (geo_a, geo_b):
        geo_ax.plot(geo_r_boundary * np.cos(geo_circle_t),
                     geo_r_boundary * np.sin(geo_circle_t),
                     color=palette['navy'], lw=1.4)
    geo_ax.set_xlim(-1.08, 1.08)
    geo_ax.set_ylim(-1.08, 1.08)
    geo_ax.grid(False)
geo_fig.suptitle('A mesh-free trial around a curved hole', fontsize=20,
                 fontweight='bold', color=palette['navy'])
save_figure(geo_fig, 'extra_annulus_solution')
""",
    },
    {
        "cell_type": "code",
        "source": r"""# Drag to orbit the annulus; the dropdown changes what the color represents.
geo_interactive_error = np.abs(geo_plot_u - geo_plot_exact)
geo_interactive = go.Figure([
    go.Surface(x=geo_plot_x, y=geo_plot_y, z=geo_plot_u,
               colorscale='Viridis', colorbar=dict(title='Neural u'),
               name='Neural solution', visible=True,
               hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>uθ=%{z:.5f}<extra></extra>'),
    go.Surface(x=geo_plot_x, y=geo_plot_y, z=geo_plot_exact,
               colorscale='Viridis', colorbar=dict(title='Exact u'),
               name='Exact solution', visible=False,
               hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>u=%{z:.5f}<extra></extra>'),
    go.Surface(x=geo_plot_x, y=geo_plot_y, z=geo_plot_u,
               surfacecolor=geo_interactive_error, customdata=geo_interactive_error,
               colorscale='Magma', colorbar=dict(title='|uθ − u|'),
               name='Error painted on the neural surface', visible=False,
               hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>uθ=%{z:.5f}'
                             '<br>|error|=%{customdata:.2e}<extra></extra>'),
])
geo_interactive.update_layout(
    title='Explore the annulus: neural solution', height=640,
    template='plotly_white', margin=dict(l=10, r=10, t=100, b=10),
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Solution value',
               aspectratio=dict(x=1, y=1, z=0.62),
               camera=dict(eye=dict(x=1.5, y=-1.6, z=1.25))),
    updatemenus=[dict(x=0.01, y=1.10, xanchor='left', yanchor='top', buttons=[
        dict(label='Neural solution', method='update',
             args=[{'visible': [True, False, False]}, {'title': 'Explore the annulus: neural solution'}]),
        dict(label='Exact solution', method='update',
             args=[{'visible': [False, True, False]}, {'title': 'Explore the annulus: exact solution'}]),
        dict(label='Error colors on neural surface', method='update',
             args=[{'visible': [False, False, True]},
                   {'title': 'Height: neural solution · Color: absolute error'}]),
    ])],
)
show_interactive(geo_interactive, 'extra_annulus_interactive')
""",
    },
    {
        "cell_type": "code",
        "source": r"""# Convergence and a radial slice give a less impressionistic accuracy check.
geo_slice_r = np.linspace(geo_a, geo_b, 220)
geo_slice_t = np.pi / 5
geo_slice_xy = torch.tensor(np.column_stack((geo_slice_r * np.cos(geo_slice_t),
                                             geo_slice_r * np.sin(geo_slice_t))), dtype=geo_dtype)
with torch.no_grad():
    geo_slice_nn = geo_trial(geo_slice_xy).numpy().ravel()
    geo_slice_ref = geo_exact(geo_slice_xy).numpy().ravel()
geo_diag_fig, geo_diag_ax = plt.subplots(1, 2, figsize=(13.5, 4.4), constrained_layout=True)
geo_diag_ax[0].semilogy(np.arange(1, len(geo_history) + 1), geo_history,
                        color=palette['teal'], lw=2.2)
geo_diag_ax[0].axhline(geo_test_res_rms**2, color=palette['orange'], ls='--',
                       lw=1.8, label='Final independent mean squared residual')
geo_diag_ax[0].set(title='Residual minimization', xlabel='Closure evaluation',
                   ylabel='Mean squared PDE residual')
geo_diag_ax[0].legend(frameon=False, fontsize=9)
geo_diag_ax[1].plot(geo_slice_r, geo_slice_ref, color=palette['navy'], lw=3,
                    label='Manufactured exact solution')
geo_diag_ax[1].plot(geo_slice_r[::9], geo_slice_nn[::9], 'o', color=palette['orange'],
                    ms=5, label='Trained neural solution')
geo_diag_ax[1].set(title=r'A radial slice at $\vartheta=\pi/5$', xlabel='Radius r', ylabel='Solution')
geo_diag_ax[1].legend(frameon=False)
save_figure(geo_diag_fig, 'extra_annulus_diagnostics')
""",
    },
    {
        "cell_type": "markdown",
        "source": r"""### Try these changes

- **Break the easy high-dimensional structure.** Replace the additive energy integrand by a function concentrated near one corner. Does 8,192-point Monte Carlo still resolve it? Track uncertainty across repeated runs, not just one estimate.
- **Make the hole larger.** Change `geo_a` to 0.7 and rerun the annulus cells. The admissible domain narrows; does the training residual still predict the solution error?
- **Test geometry sampling.** Draw the radius uniformly instead of sampling its square. Compare both models on the same independent area-uniform validation set. Which part of the domain receives disproportionate training attention?
- **Increase solution complexity.** Add angular oscillations to the manufactured solution and derive the new forcing with automatic differentiation. Increase the training point count before attributing a poor result to network capacity.

These examples show how sampling replaces a tensor grid and how a boundary factor handles a curved domain. To assess a full solver, also measure approximation error, training cost, and the work needed to represent the geometry.""",
    },
]

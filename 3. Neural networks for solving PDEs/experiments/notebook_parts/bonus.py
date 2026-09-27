"""Nonlinear reaction-diffusion bonus experiment and reproducibility outputs."""
from textwrap import dedent
CELLS = []
def md(s): CELLS.append(dict(cell_type='markdown', source=dedent(s).strip()))
def code(s): CELLS.append(dict(cell_type='code', source=dedent(s).strip()))
md(r'''
## 9 · Patterns in a reaction–diffusion system

For this experiment, we use **finite differences** to solve the Gray–Scott system.
Diffusion and reaction produce spatial patterns:
$$a_t=D_a\Delta a-ab^2+F(1-a),\qquad
b_t=D_b\Delta b+ab^2-(F+k)b.$$
We use periodic boundaries on $[0,96)^2$, grid spacing $h=1$, a five-point Laplacian,
and explicit Euler with $\Delta t=0.9$. The diffusion-only CFL limit is
$\Delta t\le h^2/(4\max(D_a,D_b))$; satisfying it does not by itself control the
nonlinear reaction error. We check a shorter run with half the time step as a diagnostic.

All three runs start from the same perturbed central patch; only $(F,k)$ changes.
The plotted height is the computed concentration $b$ on a **2D spatial domain**.

Background: [Pearson, *Complex Patterns in a Simple System* (1993)](https://doi.org/10.1126/science.261.5118.189).
''')
code('''
rd_n, rd_h, rd_dt = 96, 1., .9
rd_Da, rd_Db = .16, .08
rd_rng = np.random.default_rng(92)
rd_a0, rd_b0 = np.ones((rd_n, rd_n)), np.zeros((rd_n, rd_n))
rd_a0[39:57,39:57], rd_b0[39:57,39:57] = .5, .25
rd_a0 += .01*rd_rng.standard_normal(rd_a0.shape)
rd_b0 += .01*rd_rng.random(rd_b0.shape)

def rd_laplacian(z):
    return (np.roll(z,1,0)+np.roll(z,-1,0)+np.roll(z,1,1)+np.roll(z,-1,1)-4*z)/rd_h**2

def rd_simulate(feed, kill, steps=6500, dt=rd_dt, record=False):
    a, b = rd_a0.copy(), rd_b0.copy()
    frames, times = [b.copy()], [0.]
    for step in range(steps):
        reaction = a*b*b
        da = rd_Da*rd_laplacian(a)-reaction+feed*(1-a)
        db = rd_Db*rd_laplacian(b)+reaction-(feed+kill)*b
        a, b = a+dt*da, b+dt*db
        if record and (step+1)%130 == 0:
            frames.append(b.copy()); times.append((step+1)*dt)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a.min() > -1e-8 and b.min() > -1e-8
    return a, b, np.asarray(frames), np.asarray(times)

assert rd_dt < rd_h**2/(4*max(rd_Da, rd_Db))
rd_start = time.perf_counter()
rd_parameters = [(0.035,0.060), (0.025,0.060), (0.040,0.062)]
rd_results = [rd_simulate(f,k,record=(i==0)) for i,(f,k) in enumerate(rd_parameters)]
rd_coarse = rd_simulate(*rd_parameters[0], steps=400, dt=rd_dt)[1]
rd_fine = rd_simulate(*rd_parameters[0], steps=800, dt=rd_dt/2)[1]
rd_time_difference = np.linalg.norm(rd_coarse-rd_fine)/np.linalg.norm(rd_fine)
print(f"Three pattern simulations: {time.perf_counter()-rd_start:.1f} s")
print(f"Short-run difference at t=360, dt vs dt/2: {rd_time_difference:.2%}")
print("This time-step comparison does not estimate spatial error or certify long-time patterns.")

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), layout="constrained")
for ax, (feed,kill), (_,b,_,_) in zip(axes, rd_parameters, rd_results):
    im = ax.imshow(b, origin="lower", extent=(0,96,0,96), cmap="magma", vmin=0, vmax=.42)
    ax.set(title=f"F = {feed:.3f} · k = {kill:.3f}", xlabel="x", ylabel="y")
fig.colorbar(im, ax=axes, shrink=.72, label="Concentration b")
fig.suptitle("Same initial seed. Three different pattern geometries.", fontsize=18, weight="bold")
save_figure(fig, "09_reaction_patterns")
''')
code('''
rd_frames, rd_times = rd_results[0][2:]
rd_coords = np.arange(rd_n)[::2]
rd_X, rd_Y = np.meshgrid(rd_coords, rd_coords)
rd_fig = go.Figure(data=[go.Surface(x=rd_X, y=rd_Y, z=rd_frames[0,::2,::2],
                   colorscale="Magma", cmin=0, cmax=.42, colorbar=dict(title="b"))],
    frames=[go.Frame(name=str(i), data=[go.Surface(z=b[::2,::2])]) for i,b in enumerate(rd_frames)])
rd_fig.update_layout(title="Pattern formation · a 2D chemical field becomes a landscape",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis=dict(title="b",range=[0,.43]),
               aspectratio=dict(x=1,y=1,z=.45), uirevision="reaction"),
    updatemenus=[dict(type="buttons", direction="left", x=0, y=1.1, buttons=[
        dict(label="▶ Grow patterns",method="animate",args=[None,dict(frame=dict(duration=75,redraw=True),
             transition=dict(duration=0),fromcurrent=True)]),
        dict(label="Pause",method="animate",args=[[None],dict(mode="immediate",
             frame=dict(duration=0,redraw=False),transition=dict(duration=0))])])],
    sliders=[dict(currentvalue=dict(prefix="Time t = "), steps=[dict(label=f"{t:.0f}",method="animate",
        args=[[str(i)],dict(mode="immediate",frame=dict(duration=0,redraw=True),
        transition=dict(duration=0))]) for i,t in enumerate(rd_times)])])
show_interactive(rd_fig, "09_reaction_interactive")
''')
md(r'''
**Try:** halve the grid spacing while keeping the physical domain fixed; rescale the
Laplacian and time step consistently. Refinement changes the cost and may alter the
pattern. What would be a meaningful validation metric for a PINN trained on this system:
pointwise concentration, pattern wavelength, a time-dependent quantity of interest, or all three?

''')

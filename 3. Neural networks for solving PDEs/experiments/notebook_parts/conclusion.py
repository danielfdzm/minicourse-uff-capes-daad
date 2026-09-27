"""Notebook reproducibility report and gallery, after all experiments."""
from textwrap import dedent
CELLS = []
def md(s): CELLS.append(dict(cell_type='markdown', source=dedent(s).strip()))
def code(s): CELLS.append(dict(cell_type='code', source=dedent(s).strip()))
md(r'''
## Save the results and browse the gallery

The final cell saves the errors, software versions, and experiment settings to
`run_summary.json`. It also creates a gallery of the static and interactive plots.
Use these records to compare runs with different seeds, settings, and training budgets.
''')
code('''
# Collect diagnostics without mutating the training metric list when this cell is rerun.
summary_metrics = list(metrics)
for inv_name, inv_result in inv_results.items():
    inv_rel = float(np.linalg.norm(inv_result["prediction"]-inv_sine/inv_kappa_true)
                    / np.linalg.norm(inv_sine/inv_kappa_true))
    summary_metrics.append(dict(experiment=f"Inverse: {inv_name}", relative_l2=inv_rel,
                        kappa=inv_result["kappa_nn"], kappa_exact_model=inv_result["kappa_ls"]))
summary_metrics.append(dict(experiment="Annulus PINN", relative_l2=geo_relative_l2,
                    residual_rms=geo_test_res_rms, seconds=geo_seconds))
report = dict(versions=versions, run_mode=RUN_MODE, seed=SEED, budgets=BUDGET,
              device="cpu", dtype=str(torch.get_default_dtype()),
              local_seeds=dict(forward=SEED, inverse=23, high_dimension=1701,
                               annulus=1702, reaction_diffusion=92),
              additional_settings=dict(inverse_adam=1000, inverse_lbfgs=140,
                  inverse_sensors=inv_count, inverse_noise_sd=inv_sigma,
                  annulus_lbfgs=130, annulus_points=512, high_dim_repeats=hd_repeats,
                  reaction_steps=6500, reaction_dt=rd_dt, reaction_grid=rd_n,
                  reaction_parameters=rd_parameters),
              expanded_experiments=dict(multiscale=MULTISCALE_SETTINGS,
                  adaptive=ADAPTIVE_SETTINGS, heat=HEAT_SETTINGS, parametric=PARAMETRIC_SETTINGS),
              static_figures=FIGURE_NAMES, interactive_figures=INTERACTIVE_NAMES,
              metrics=summary_metrics, elapsed_seconds=time.perf_counter()-NOTEBOOK_START,
              wave_relative_energy_drift=float(np.ptp(wave_E)/wave_E_exact),
              reaction_time_step_difference=float(rd_time_difference))
(OUTPUT_DIR / "run_summary.json").write_text(json.dumps(report, indent=2))
np.savez_compressed(OUTPUT_DIR / "computed_fields.npz", poisson_x=p2_axis,
    poisson_prediction=p2_pred, poisson_exact=p2_exact, poisson_residual=p2_r,
    boundary_weights=p1_lambdas, boundary_diagnostics=p1_sweep,
    annulus_x=geo_plot_x, annulus_y=geo_plot_y, annulus_prediction=geo_plot_u,
    annulus_exact=geo_plot_exact, reaction_b=np.stack([r[1] for r in rd_results]),
    multiscale_x=ms_test_x, multiscale_exact=ms_reference,
    multiscale_plain=ms_results["Plain tanh"]["prediction"],
    multiscale_fourier=ms_results["Fourier features"]["prediction"],
    heat_x=ht_x, heat_t=ht_t, heat_prediction=ht_prediction,
    heat_exact=ht_exact, heat_residual=ht_test_residual,
    adaptive_x=ad_x, adaptive_exact=ad_exact,
    adaptive_labels=np.array(list(ad_results)),
    adaptive_predictions=np.stack([result["prediction"] for result in ad_results.values()]),
    adaptive_residuals=np.stack([result["residual"] for result in ad_results.values()]),
    parametric_x=pa_x, parametric_mu=pa_mu, parametric_prediction=pa_prediction,
    parametric_exact=pa_exact, parametric_fd_prediction=pa_fd_predictions)

from html import escape
cards = []
for name in FIGURE_NAMES:
    png = OUTPUT_DIR / f"{name}.png"
    title = png.stem.replace("_", " ").title()
    cards.append(f'<a class="card" href="{escape(png.name)}"><img src="{escape(png.name)}" '
                 f'alt="{escape(title)}"><h3>{escape(title)}</h3></a>')
interactive_links = "".join(f'<a class="pill" href="{escape(name)}.html">{escape(name.replace("_", " ").title())} ↗</a>'
                            for name in INTERACTIVE_NAMES)
gallery_html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural PDE Laboratory · Visual Gallery</title>
<style>body{margin:0;background:#0d1f32;color:#e9f4fc;font-family:system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:50px 28px}h1{font-size:clamp(34px,5vw,62px);letter-spacing:-2px;margin:8px 0}
p{color:#b6cfdd;line-height:1.65;max-width:760px}.eyebrow{color:#47d9c2;letter-spacing:3px;text-transform:uppercase;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px;margin-top:30px}
.card{background:#173049;border:1px solid #29445c;border-radius:16px;overflow:hidden;text-decoration:none;color:inherit;transition:transform .2s}
.card:hover{transform:translateY(-4px)}.card img{width:100%;display:block;background:#f7fafc}.card h3{font-size:16px;padding:2px 18px 15px}
.pill{display:inline-block;background:#176f78;color:white;padding:12px 16px;border-radius:30px;text-decoration:none;margin:6px 8px 6px 0;font-size:14px}
</style><main><div class="eyebrow">Computed experiments · 2D / 3D · Offline</div>
<h1>Neural PDE Laboratory</h1><p>Browse the notebook results: membrane motion, neural PDE solutions, sensor placement,
and reaction–diffusion patterns. Open an interactive plot to rotate the surface,
play an animation, or inspect individual values.</p>"""
gallery_html += interactive_links + '<div class="grid">' + "".join(cards) + '</div></main></html>'
(OUTPUT_DIR / "index.html").write_text(gallery_html)
print(f"Completed in {report['elapsed_seconds']:.1f} seconds.")
print(f"Saved {len(cards)} static figure sets, {len(INTERACTIVE_NAMES)} offline interactive explorers, computed arrays, and run_summary.json.")
print(f"Open {OUTPUT_DIR / 'index.html'} for the visual gallery.")
for row in summary_metrics:
    print(f"{row['experiment']:32s} relative L2 = {row['relative_l2']:.3e}")
''')

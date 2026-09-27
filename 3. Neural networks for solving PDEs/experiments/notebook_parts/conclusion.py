"""Part-specific reports: each notebook exports only its own computed results."""
from textwrap import dedent

# Expressions are embedded as notebook source, so no source-module import is needed.
FIELDS = {
    '01_foundations': '''boundary_weights=p1_lambdas, boundary_diagnostics=p1_sweep,
        wave_times=wave_times, wave_energy=wave_E,
        poisson_x=p1_grid, poisson_exact=p1_exact,
        pinn_prediction=p1_results["Hard PINN"][0],
        ritz_prediction=p1_results["Deep Ritz"][0]''',
    '02_poisson_and_limits': '''poisson_x=p2_axis, poisson_prediction=p2_pred,
        poisson_exact=p2_exact, poisson_residual=p2_r,
        counterexample_n=ce_n, counterexample_loss=ce_J,
        constant_sequence_loss=ce_J_constant''',
    '03_inverse': '''inverse_x=inv_grid, inverse_reference=inv_sine/inv_kappa_true,
        inverse_predictions=np.stack([r["prediction"] for r in inv_results.values()]),
        inverse_labels=np.array(list(inv_results))''',
    '04_dimensions_and_geometry': '''annulus_x=geo_plot_x, annulus_y=geo_plot_y,
        annulus_prediction=geo_plot_u, annulus_exact=geo_plot_exact''',
    '05_reaction_diffusion': '''reaction_b=np.stack([r[1] for r in rd_results])''',
    '06_features_and_sampling': '''multiscale_x=ms_test_x, multiscale_exact=ms_reference,
        multiscale_plain=ms_results["Plain tanh"]["prediction"],
        multiscale_fourier=ms_results["Fourier features"]["prediction"],
        adaptive_x=ad_x, adaptive_exact=ad_exact,
        adaptive_labels=np.array(list(ad_results)),
        adaptive_predictions=np.stack([r["prediction"] for r in ad_results.values()]),
        adaptive_residuals=np.stack([r["residual"] for r in ad_results.values()])''',
    '07_time_and_parameters': '''heat_x=ht_x, heat_t=ht_t, heat_prediction=ht_prediction,
        heat_exact=ht_exact, heat_residual=ht_test_residual,
        parametric_x=pa_x, parametric_mu=pa_mu, parametric_prediction=pa_prediction,
        parametric_exact=pa_exact, parametric_fd_prediction=pa_fd_predictions''',
}
SETTINGS = {
    '01_foundations': 'dict(wave_relative_energy_drift=float(np.ptp(wave_E)/wave_E_exact))',
    '02_poisson_and_limits': 'dict(poisson_collocation=768)',
    '03_inverse': 'dict(seed=23, adam=1000, lbfgs=140, sensors=inv_count, noise_sd=inv_sigma)',
    '04_dimensions_and_geometry': 'dict(high_dimension_seed=1701, annulus_seed=1702, annulus_lbfgs=130, annulus_points=512, high_dim_repeats=hd_repeats)',
    '05_reaction_diffusion': 'dict(seed=92, steps=6500, dt=rd_dt, grid=rd_n, parameters=rd_parameters, time_step_difference=float(rd_time_difference))',
    '06_features_and_sampling': 'dict(multiscale=MULTISCALE_SETTINGS, adaptive=ADAPTIVE_SETTINGS)',
    '07_time_and_parameters': 'dict(heat=HEAT_SETTINGS, parametric=PARAMETRIC_SETTINGS)',
}


def make_cells(slug):
    extra = ''
    if slug == '03_inverse':
        extra = dedent('''
            for name, result in inv_results.items():
                rel = float(np.linalg.norm(result["prediction"]-inv_sine/inv_kappa_true)
                            / np.linalg.norm(inv_sine/inv_kappa_true))
                summary_metrics.append(dict(experiment=f"Inverse: {name}", relative_l2=rel,
                    kappa=result["kappa_nn"], kappa_exact_model=result["kappa_ls"]))
        ''')
    elif slug == '04_dimensions_and_geometry':
        extra = '''summary_metrics.append(dict(experiment="Annulus PINN", relative_l2=geo_relative_l2,
    residual_rms=geo_test_res_rms, seconds=geo_seconds))\n'''
    source = f'''# Keep reports separate so running one notebook cannot replace another's results.
PART = {slug!r}
summary_metrics = list(metrics)
{extra}
report = dict(part=PART, versions=versions, run_mode=RUN_MODE, seed=SEED, budgets=BUDGET,
    device="cpu", dtype=str(torch.get_default_dtype()), settings={SETTINGS[slug]},
    static_figures=FIGURE_NAMES, interactive_figures=INTERACTIVE_NAMES,
    metrics=summary_metrics, elapsed_seconds=time.perf_counter()-NOTEBOOK_START)
(OUTPUT_DIR / f"run_summary_{{PART}}.json").write_text(json.dumps(report, indent=2))
np.savez_compressed(OUTPUT_DIR / f"computed_fields_{{PART}}.npz", {FIELDS[slug]})

from html import escape
cards = []
for name in FIGURE_NAMES:
    title = escape(name.replace("_", " ").title())
    cards.append(f'<figure><a href="{{name}}.png"><img src="{{name}}.png" alt="{{title}}"></a>'
                 f'<figcaption>{{title}} · <a href="{{name}}.pdf">PDF</a></figcaption></figure>')
links = "".join(f'<li><a href="{{name}}.html">{{escape(name.replace("_", " ").title())}}</a></li>'
                for name in INTERACTIVE_NAMES)
page = ('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Neural PDE Laboratory</title><style>'
        'body{{font-family:system-ui;color:#273442;max-width:1100px;margin:auto;padding:32px}}'
        'img{{max-width:100%}}figure{{margin:32px 0}}a{{color:#4d708a}}</style>'
        f'<h1>Neural PDE Laboratory · {{escape(PART)}}</h1><ul>' + links + '</ul>' + "".join(cards) + '</html>')
(OUTPUT_DIR / f"index_{{PART}}.html").write_text(page)
print(f"Completed in {{report['elapsed_seconds']:.1f}} seconds.")
print(f"Saved {{len(FIGURE_NAMES)}} figure sets and {{len(INTERACTIVE_NAMES)}} interactive explorers.")
print(f"Reports: run_summary_{{PART}}.json, computed_fields_{{PART}}.npz, index_{{PART}}.html")
for row in summary_metrics:
    print(f"{{row['experiment']:32s}} relative L2 = {{row['relative_l2']:.3e}}")
'''
    return [dict(cell_type='markdown', source='''## Save this notebook's results

This cell records versions, settings, diagnostics and computed fields, and builds a
local gallery. Report filenames include this notebook's part number, so notebooks
can run independently in any order without overwriting each other's reports.'''),
            dict(cell_type='code', source=source)]

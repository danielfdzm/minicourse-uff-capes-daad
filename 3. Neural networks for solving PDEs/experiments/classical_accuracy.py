"""Classical accuracy companions for the saved one-dimensional PINN example.

Run ``python3 experiments/classical_accuracy.py`` after ``forward_1d.py``.
Both solvers use only the forcing and zero Dirichlet values. The exact solution
enters diagnostics only. All methods are evaluated on the same independently
chosen 1001-point grid; some classical nodes coincide with evaluation points.
This smooth manufactured problem lets us compare accuracy. Runtime and
degrees of freedom are not matched.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent / "results"
FD_INTERVALS = [16, 32, 64, 128, 256, 512, 1024]
SPECTRAL_DEGREES = [8, 12, 16, 24, 32, 48]


def forcing(x):
    return np.pi**2 * np.sin(np.pi * x)


def tridiagonal_solve(lower, diagonal, upper, rhs):
    """Thomas elimination; inputs are copied, with no explicit matrix inverse."""
    a, b, c, d = [np.array(v, dtype=np.float64, copy=True)
                  for v in (lower, diagonal, upper, rhs)]
    for i in range(1, len(b)):
        multiplier = a[i - 1] / b[i - 1]
        b[i] -= multiplier * c[i - 1]
        d[i] -= multiplier * d[i - 1]
    solution = np.empty_like(d)
    solution[-1] = d[-1] / b[-1]
    for i in range(len(b) - 2, -1, -1):
        solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i]
    return solution


def error_metrics(prediction, grid):
    exact = np.sin(np.pi * grid)
    difference = prediction - exact
    return {
        "relative_l2": float(np.sqrt(np.trapezoid(difference**2, grid)
                                     / np.trapezoid(exact**2, grid))),
        "max_abs_error": float(np.max(np.abs(difference))),
    }


def residual_metrics(residual, rhs, matrix_norm_inf, solution):
    max_residual = np.linalg.norm(residual, ord=np.inf)
    return {
        "nodal_residual_max_abs": float(max_residual),
        "nodal_residual_relative_inf": float(max_residual / np.linalg.norm(rhs, ord=np.inf)),
        "backward_error_inf": float(max_residual / (
            matrix_norm_inf * np.linalg.norm(solution, ord=np.inf)
            + np.linalg.norm(rhs, ord=np.inf))),
    }


def finite_difference(intervals, grid):
    # BEGIN FD_SOLVE
    x = np.linspace(0.0, 1.0, intervals + 1, dtype=np.float64)
    h = 1.0 / intervals
    interior_count = intervals - 1
    u = np.zeros(intervals + 1, dtype=np.float64)
    u[1:-1] = tridiagonal_solve(
        -np.ones(interior_count - 1), 2 * np.ones(interior_count),
        -np.ones(interior_count - 1), h**2 * forcing(x[1:-1]))
    prediction = np.interp(grid, x, u)  # Piecewise linear interpolation.
    # END FD_SOLVE
    rhs = forcing(x[1:-1])
    residual = (2 * u[1:-1] - u[:-2] - u[2:]) / h**2 - rhs
    metrics = {
        "intervals": intervals, "node_count": intervals + 1,
        "free_dof": interior_count,
        **error_metrics(prediction, grid),
        **residual_metrics(residual, rhs, 4 / h**2, u[1:-1]),
    }
    return prediction, metrics, x, u, residual


def barycentric_lobatto(nodes, values, grid):
    """Evaluate the global interpolating polynomial on Chebyshev--Lobatto nodes."""
    weights = (-1.0) ** np.arange(len(nodes))
    weights[[0, -1]] *= 0.5
    differences = grid[:, None] - nodes[None, :]
    matches = differences == 0.0
    at_node = matches.any(axis=1)
    prediction = np.empty_like(grid)
    ratios = weights[None, :] / differences[~at_node]
    prediction[~at_node] = (ratios @ values) / ratios.sum(axis=1)
    prediction[at_node] = values[np.argmax(matches[at_node], axis=1)]
    return prediction


def chebyshev_collocation(degree, grid):
    # Chebyshev differentiation matrix in descending t = cos(j*pi/N) order.
    t = np.cos(np.pi * np.arange(degree + 1, dtype=np.float64) / degree)
    x = (t + 1.0) / 2.0
    c = (-1.0) ** np.arange(degree + 1)
    c[[0, -1]] *= 2.0
    differences = t[:, None] - t[None, :]
    derivative = (c[:, None] / c[None, :]) / (differences + np.eye(degree + 1))
    derivative -= np.diag(derivative.sum(axis=1))
    # d/dx = 2 d/dt, hence -d^2/dx^2 = -4 D^2.
    operator = -4.0 * (derivative @ derivative)
    interior_operator = operator[1:-1, 1:-1]
    rhs = forcing(x[1:-1])
    u = np.zeros(degree + 1, dtype=np.float64)
    u[1:-1] = np.linalg.solve(interior_operator, rhs)
    prediction = barycentric_lobatto(x, u, grid)
    residual = interior_operator @ u[1:-1] - rhs
    metrics = {
        "degree": degree, "node_count": degree + 1, "free_dof": degree - 1,
        **error_metrics(prediction, grid),
        **residual_metrics(residual, rhs, np.linalg.norm(interior_operator, ord=np.inf),
                           u[1:-1]),
    }
    return prediction, metrics, x, u, residual


def main():
    forward_path = OUT / "forward_1d.npz"
    forward_report_path = OUT / "forward_1d.json"
    with np.load(forward_path) as forward:
        grid = forward["x"].astype(np.float64, copy=True)
        pinn_u = forward["hard_pinn_u"].copy()
    forward_report = json.loads(forward_report_path.read_text())
    assert np.array_equal(grid, np.linspace(0.0, 1.0, 1001))
    pinn_metrics = error_metrics(pinn_u, grid)
    assert np.isclose(pinn_metrics["relative_l2"],
                      forward_report["main_models"]["hard_pinn"]["final"]["relative_l2"],
                      rtol=1e-12, atol=0.0)
    arrays = {
        "x": grid, "exact_u": np.sin(np.pi * grid),
        "float64_eps": np.asarray(np.finfo(np.float64).eps),
        "pinn_u": pinn_u,
        "pinn_relative_l2": np.asarray(pinn_metrics["relative_l2"]),
    }
    methods = {}
    for name, solver, resolutions in [
            ("fd", finite_difference, FD_INTERVALS),
            ("spectral", chebyshev_collocation, SPECTRAL_DEGREES)]:
        runs = [solver(resolution, grid) for resolution in resolutions]
        methods[name] = [run[1] for run in runs]
        arrays[f"{name}_u"] = np.stack([run[0] for run in runs])
        for field in methods[name][0]:
            arrays[f"{name}_{field}"] = np.asarray([run[field] for run in methods[name]])
        # Preserve full nodal solutions and residual vectors without object arrays.
        for resolution, (_, _, x, u, residual) in zip(resolutions, runs):
            arrays[f"{name}_{resolution}_nodes"] = x
            arrays[f"{name}_{resolution}_nodal_u"] = u
            arrays[f"{name}_{resolution}_nodal_residual"] = residual

    fd_ratios = arrays["fd_relative_l2"][:-1] / arrays["fd_relative_l2"][1:]
    arrays["fd_error_reduction_on_doubling"] = fd_ratios
    assert all(np.all(np.isfinite(array)) for array in arrays.values())
    assert np.all((fd_ratios[:3] > 3.8) & (fd_ratios[:3] < 4.2)), fd_ratios
    assert np.min(arrays["spectral_relative_l2"]) < 1e-10
    assert all(np.max(np.abs(arrays[f"{name}_{resolution}_nodal_u"][[0, -1]])) == 0
               for name, resolutions in [("fd", FD_INTERVALS), ("spectral", SPECTRAL_DEGREES)]
               for resolution in resolutions)

    report = {
        "problem": "-u'' = pi^2 sin(pi x), 0 < x < 1; u(0)=u(1)=0",
        "exact_solution_for_diagnostics_only": "sin(pi x)",
        "config": {
            "dtype": "float64", "float64_eps": float(np.finfo(np.float64).eps),
            "diagnostic_grid_points": len(grid),
            "diagnostic_grid": "Same 1001-point uniform grid saved in forward_1d.npz",
            "relative_l2_integration": "Composite trapezoidal rule on diagnostic grid",
            "fd": {
                "intervals": FD_INTERVALS, "scheme": "Second-order centered differences",
                "linear_solver": "Thomas tridiagonal elimination",
                "evaluation": "Piecewise linear interpolation of nodal solution (numpy.interp)",
            },
            "spectral": {
                "degrees": SPECTRAL_DEGREES, "scheme": "Chebyshev-Lobatto polynomial collocation",
                "linear_solver": "numpy.linalg.solve on interior differentiation matrix",
                "evaluation": "Barycentric polynomial interpolation with Chebyshev-Lobatto weights",
            },
        },
        "provenance": {
            "command": "python3 experiments/classical_accuracy.py",
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "forward_npz_sha256": hashlib.sha256(forward_path.read_bytes()).hexdigest(),
            "forward_json_sha256": hashlib.sha256(forward_report_path.read_bytes()).hexdigest(),
        },
        "methods": methods,
        "saved_pinn_reference": {
            **pinn_metrics,
            "description": "Existing seed-7 hard-boundary PINN after Adam and L-BFGS",
            "parameter_count": forward_report["config"]["parameter_count"],
            "training_nodes": forward_report["config"]["quadrature_nodes"],
        },
        "validation": {
            "all_numeric_arrays_finite": True,
            "fd_error_reduction_on_doubling": fd_ratios.tolist(),
            "fd_first_three_ratios_between_3_8_and_4_2": True,
            "best_spectral_relative_l2_below_1e_minus_10": True,
            "classical_nodal_boundaries_exactly_zero": True,
            "solver_uses_exact_solution_labels": False,
        },
        "notes": [
            "This smooth, one-dimensional manufactured solution favors polynomial spectral methods.",
            "The diagnostic grid is chosen independently of each solve, but some classical nodes coincide with it.",
            "Free DOF excludes the two prescribed endpoint values; node count includes both endpoints.",
            "Nodal residuals are algebraic PDE residuals at interior solver nodes, not continuous residual norms.",
            "Relative residual = ||A u - f||_inf / ||f||_inf; normwise backward error divides by ||A||_inf ||u||_inf + ||f||_inf.",
            "FD interpolation contributes to the reported solution error; linear interpolation retains second-order accuracy here.",
            "Spectral errors need not decrease monotonically at high degree because floating-point roundoff and conditioning matter.",
            "Machine epsilon is floating-point spacing near 1, not a guaranteed PDE solution error.",
            "The saved PINN is one configuration and seed; this is not a universal method ranking or equal-cost/DOF comparison.",
            "No timings are compared. A small algebraic residual does not remove discretization or interpolation error.",
        ],
    }
    np.savez_compressed(OUT / "classical_accuracy.npz", **arrays)
    (OUT / "classical_accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in methods:
        for metrics in methods[name]:
            print(f"{name:8s} nodes={metrics['node_count']:4d} "
                  f"rel.L2={metrics['relative_l2']:.3e} "
                  f"residual={metrics['nodal_residual_max_abs']:.3e}")
    print(f"FD doubling ratios: {fd_ratios}")
    print(f"Saved {OUT / 'classical_accuracy.npz'}")


if __name__ == "__main__":
    main()

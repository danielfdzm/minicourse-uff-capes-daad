"""Run the Poisson experiments used in the lecture slides on CPU.

Run from the project root with ``python3 experiments/forward_1d.py``.
Only the forcing f and the boundary conditions enter training. The exact
solution is used to check errors after training.

All methods start from an identical seeded 1-32-32-1 tanh network. A normalized
residual, (-u'' - f) / pi**2, makes the soft-penalty sweep interpretable. The
PINN and Deep Ritz examples receive Adam plus L-BFGS; the soft-boundary sweep
and its hard-boundary baseline receive the same Adam-only budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


SEED = 7
ADAM_STEPS = 2000
LEARNING_RATE = 1e-3
QUADRATURE_NODES = 128
LBFGS_STEPS = 300
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
OUT = Path(__file__).resolve().parent / "results"


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Autodifferentiate a scalar network output at independent sample points."""
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def forcing(x: torch.Tensor) -> torch.Tensor:
    return math.pi**2 * torch.sin(math.pi * x)


class Solution(nn.Module):
    def __init__(self, initial_network: nn.Module, hard_boundary: bool):
        super().__init__()
        self.net = copy.deepcopy(initial_network)
        self.hard_boundary = hard_boundary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return x * (1.0 - x) * raw if self.hard_boundary else raw


# BEGIN PINN_LOSS
def pinn_loss(model, x, weights):
    u = model(x)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)
    residual = (-u_xx - forcing(x)) / math.pi**2
    return (weights * residual.square()).sum()
# END PINN_LOSS


# BEGIN RITZ_LOSS
def ritz_loss(model, x, weights):
    u = model(x)
    u_x = derivative(u, x)
    return (weights * (0.5 * u_x.square() - forcing(x) * u)).sum()
# END RITZ_LOSS


# BEGIN SOFT_LOSS
def soft_loss(model, x, weights, penalty):
    boundary = torch.tensor([[0.0], [1.0]])
    boundary_loss = model(boundary).square().mean()
    return pinn_loss(model, x, weights) + penalty * boundary_loss
# END SOFT_LOSS


def evaluate(model, grid):
    """Report errors on a dense grid absent from the training quadrature."""
    x = torch.tensor(grid[:, None], requires_grad=True)
    u = model(x)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)
    predicted = u.detach().numpy().ravel()
    predicted_derivative = u_x.detach().numpy().ravel()
    residual = (-u_xx - forcing(x)).detach().numpy().ravel()
    exact = np.sin(np.pi * grid)
    exact_derivative = np.pi * np.cos(np.pi * grid)
    l2_squared = np.trapezoid((predicted - exact) ** 2, grid)
    h1_squared = np.trapezoid((predicted_derivative - exact_derivative) ** 2, grid)
    metrics = {
        "relative_l2": float(np.sqrt(l2_squared / np.trapezoid(exact**2, grid))),
        "relative_h1_seminorm": float(
            np.sqrt(h1_squared / np.trapezoid(exact_derivative**2, grid))
        ),
        "boundary_max_abs": float(np.max(np.abs(predicted[[0, -1]]))),
        "physical_residual_rms": float(np.sqrt(np.trapezoid(residual**2, grid))),
        "normalized_residual_rms": float(
            np.sqrt(np.trapezoid(residual**2, grid)) / np.pi**2
        ),
    }
    if model.hard_boundary:
        # For a conforming approximation, E(v)-E(u*) = ||v'-u*'||_L2^2 / 2.
        # Evaluate the positive identity to avoid subtractive cancellation.
        metrics["energy_gap_from_error_identity"] = float(0.5 * h1_squared)
    return {
        "u": predicted,
        "du": predicted_derivative,
        "residual": residual,
        "metrics": metrics,
    }


def train(model, objective, grid, refine=False, label="model"):
    start = time.perf_counter()
    history = []

    def record(step, stage):
        metrics = evaluate(model, grid)["metrics"]
        history.append({"step": step, "stage": stage, "loss": float(objective().detach()), **metrics})

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    record(0, "initial")
    for step in range(1, ADAM_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            record(step, "adam")

    adam_result = evaluate(model, grid)
    adam_parameters = copy.deepcopy(model.state_dict())
    lbfgs_iterations = 0
    lbfgs_evaluations = 0
    if refine:
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=25, max_eval=40,
            tolerance_grad=1e-11, tolerance_change=1e-14,
            history_size=50, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        for _ in range(LBFGS_STEPS // 25):
            optimizer.step(closure)
            state = optimizer.state[next(iter(model.parameters()))]
            lbfgs_iterations = int(state["n_iter"])
            lbfgs_evaluations = int(state["func_evals"])
            record(ADAM_STEPS + lbfgs_iterations, "lbfgs")

    result = evaluate(model, grid)
    result.update({
        "history": history,
        "adam_result": adam_result,
        "adam_parameters": adam_parameters,
        "runtime_seconds": time.perf_counter() - start,
        "lbfgs_iterations": lbfgs_iterations,
        "lbfgs_function_evaluations": lbfgs_evaluations,
    })
    print(f"{label:15s}: rel. L2={result['metrics']['relative_l2']:.3e}; "
          f"rel. H1={result['metrics']['relative_h1_seminorm']:.3e}; "
          f"BC={result['metrics']['boundary_max_abs']:.3e}; "
          f"time={result['runtime_seconds']:.1f}s", flush=True)
    return result


def main():
    start = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    initial = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
    )
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(QUADRATURE_NODES)
    x = torch.tensor(((nodes + 1.0) / 2.0)[:, None], requires_grad=True)
    weights = torch.tensor((quadrature_weights / 2.0)[:, None])
    grid = np.linspace(0.0, 1.0, 1001)

    hard_model = Solution(initial, hard_boundary=True)
    hard = train(hard_model, lambda: pinn_loss(hard_model, x, weights), grid,
                 refine=True, label="Hard PINN")
    ritz_model = Solution(initial, hard_boundary=True)
    ritz = train(ritz_model, lambda: ritz_loss(ritz_model, x, weights), grid,
                 refine=True, label="Deep Ritz")

    soft_results = []
    for penalty in LAMBDAS:
        model = Solution(initial, hard_boundary=False)
        result = train(model, lambda: soft_loss(model, x, weights, penalty), grid,
                       label=f"Soft {penalty:g}")
        soft_results.append(result)

    arrays = {
        "x": grid,
        "exact_u": np.sin(np.pi * grid),
        "exact_du": np.pi * np.cos(np.pi * grid),
        "quadrature_x": x.detach().numpy().ravel(),
        "quadrature_weights": weights.numpy().ravel(),
        "soft_lambda": np.asarray(LAMBDAS),
        "soft_u": np.stack([result["u"] for result in soft_results]),
        "soft_du": np.stack([result["du"] for result in soft_results]),
        "soft_residual": np.stack([result["residual"] for result in soft_results]),
    }
    for name, result in [("hard_pinn", hard), ("ritz", ritz)]:
        for field in ["u", "du", "residual"]:
            arrays[f"{name}_{field}"] = result[field]
            arrays[f"{name}_adam_{field}"] = result["adam_result"][field]
        for field in ["step", "loss", "relative_l2", "relative_h1_seminorm",
                      "physical_residual_rms", "energy_gap_from_error_identity"]:
            arrays[f"{name}_history_{field}"] = np.asarray([r[field] for r in result["history"]])
        arrays[f"{name}_history_stage"] = np.asarray([r["stage"] for r in result["history"]])
    for field in ["relative_l2", "relative_h1_seminorm", "boundary_max_abs",
                  "physical_residual_rms", "normalized_residual_rms"]:
        arrays[f"soft_{field}"] = np.asarray([r["metrics"][field] for r in soft_results])
        arrays[f"hard_adam_{field}"] = np.asarray(hard["adam_result"]["metrics"][field])

    assert all(np.all(np.isfinite(a)) for a in arrays.values() if a.dtype.kind != "U")
    assert hard["metrics"]["relative_l2"] < 1e-3, "Hard PINN did not converge."
    assert ritz["metrics"]["relative_l2"] < 1e-3, "Deep Ritz did not converge."
    assert hard["metrics"]["boundary_max_abs"] == 0.0
    assert ritz["metrics"]["boundary_max_abs"] == 0.0

    report = {
        "problem": "-u'' = pi^2 sin(pi x), 0 < x < 1; u(0)=u(1)=0",
        "exact_solution_for_diagnostics_only": "sin(pi x)",
        "config": {
            "seed": SEED, "architecture": [1, 32, 32, 1], "activation": "tanh",
            "parameter_count": sum(p.numel() for p in initial.parameters()),
            "dtype": "float64", "device": "cpu", "threads": 1,
            "adam_steps": ADAM_STEPS, "adam_learning_rate": LEARNING_RATE,
            "lbfgs_max_iterations_main_models": LBFGS_STEPS,
            "lbfgs_line_search": "strong_wolfe",
            "quadrature": "Gauss-Legendre on [0,1]", "quadrature_nodes": QUADRATURE_NODES,
            "diagnostic_grid_points": len(grid),
            "pinn_residual": "(-u_xx - f) / pi^2",
            "soft_boundary_loss": "lambda * mean([u(0)^2, u(1)^2])",
            "hard_boundary_ansatz": "x * (1-x) * net(x)",
            "soft_lambdas": LAMBDAS,
            "initialization": "Identical deepcopy of the seed-7 initial network for every run",
            "sweep_budget": "2000 Adam steps for every soft model and the hard baseline; no L-BFGS",
        },
        "provenance": {
            "command": "python3 experiments/forward_1d.py",
            "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "main_models": {},
        "boundary_sweep": {
            "hard_adam_baseline": hard["adam_result"]["metrics"],
            "soft": [{"lambda": penalty, **result["metrics"],
                      "runtime_seconds": result["runtime_seconds"]}
                     for penalty, result in zip(LAMBDAS, soft_results)],
        },
        "validation": {
            "all_numeric_arrays_finite": True,
            "main_model_relative_l2_below_1e_minus_3": True,
            "hard_boundaries_exactly_zero": True,
            "training_uses_exact_solution_labels": False,
        },
        "notes": [
            "One seed is a demonstration, not a statistical method ranking.",
            "Losses have different units; compare diagnostic errors, not objective values.",
            "Energy gap identity applies only to the hard-boundary (conforming) models.",
            "Diagnostic norms use composite trapezoidal integration on 1001 independent grid points.",
        ],
        "total_runtime_seconds": time.perf_counter() - start,
    }
    for name, result in [("hard_pinn", hard), ("ritz", ritz)]:
        report["main_models"][name] = {
            "final": result["metrics"], "after_adam": result["adam_result"]["metrics"],
            "history": result["history"], "runtime_seconds": result["runtime_seconds"],
            "lbfgs_iterations": result["lbfgs_iterations"],
            "lbfgs_function_evaluations": result["lbfgs_function_evaluations"],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "forward_1d.npz", **arrays)
    (OUT / "forward_1d.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {OUT / 'forward_1d.npz'}", flush=True)


if __name__ == "__main__":
    main()

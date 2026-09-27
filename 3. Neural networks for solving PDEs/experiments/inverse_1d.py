#!/usr/bin/env python3
"""Infer a positive diffusion coefficient from the PDE and 12 noisy sensors.

Run ``python3 experiments/inverse_1d.py`` from the chapter folder. This CPU
experiment compares hybrid (physics + data) training with physics-only
training from the same initial parameters.  The latter is not an
identifiable inverse problem: every k > 0 admits u(x) = sin(pi*x)/k.

Exact values are used only to generate synthetic sensor observations and for
evaluation.  Network training never receives dense exact solution labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEED = 23
TRUE_K = 0.7
INITIAL_K = 1.3
SENSOR_COUNT = 12
DATA_WEIGHT = 5.0
EPSILON = 1.0e-6


class InversePINN(nn.Module):
    """A smooth neural trial function with both boundary values built in."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        # BEGIN positivity
        # Optimize an unconstrained scalar; the physical coefficient stays > 0.
        initial_raw = np.log(np.expm1(INITIAL_K - EPSILON))
        self.raw_k = nn.Parameter(torch.tensor(initial_raw))

    @property
    def k(self) -> torch.Tensor:
        return F.softplus(self.raw_k) + EPSILON
        # END positivity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - x) * self.net(x)


def normalized_residual(model: InversePINN, x: torch.Tensor) -> torch.Tensor:
    """Autodifferentiate u twice and divide the PDE residual by pi**2."""
    u = model(x)
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]
    return -model.k * d2u / torch.pi**2 - torch.sin(torch.pi * x)


def loss_terms(
    model: InversePINN,
    x_r: torch.Tensor,
    x_obs: torch.Tensor,
    y_obs: torch.Tensor,
    data_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # BEGIN inverse_loss
    r = normalized_residual(model, x_r)
    loss_physics = r.square().mean()
    loss_data = (model(x_obs) - y_obs).square().mean()
    loss = loss_physics + data_weight * loss_data
    # END inverse_loss
    return loss, loss_physics, loss_data


def train(
    model: InversePINN,
    x_r: torch.Tensor,
    x_obs: torch.Tensor,
    y_obs: torch.Tensor,
    data_weight: float,
    adam_steps: int,
    lbfgs_steps: int,
) -> dict[str, np.ndarray]:
    """Record every 20 Adam updates and each completed L-BFGS outer step."""
    history: list[tuple[float, ...]] = []
    n_evals = 0
    start = time.perf_counter()

    def record(step: int, phase: int) -> None:
        total, physics, data = loss_terms(model, x_r, x_obs, y_obs, data_weight)
        history.append((step, phase, n_evals, float(model.k.detach()),
                        float(total.detach()), float(physics.detach()),
                        float(data.detach())))

    record(0, 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    for step in range(1, adam_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        x_r.grad = None
        loss, _, _ = loss_terms(model, x_r, x_obs, y_obs, data_weight)
        loss.backward()
        optimizer.step()
        n_evals += 1
        if step % 20 == 0 or step == adam_steps:
            record(step, 0)

    # Five inner iterations per outer call keep a useful, inexpensive trajectory.
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=5, max_eval=8,
        tolerance_grad=1.0e-11, tolerance_change=1.0e-13,
        history_size=50, line_search_fn="strong_wolfe",
    )
    for outer_step in range(1, lbfgs_steps + 1):
        def closure() -> torch.Tensor:
            nonlocal n_evals
            optimizer.zero_grad(set_to_none=True)
            x_r.grad = None
            total, _, _ = loss_terms(model, x_r, x_obs, y_obs, data_weight)
            total.backward()
            n_evals += 1
            return total

        optimizer.step(closure)
        record(adam_steps + outer_step, 1)

    elapsed = time.perf_counter() - start
    values = np.asarray(history)
    return {
        "step": values[:, 0].astype(np.int64),
        "phase": values[:, 1].astype(np.int64),
        "loss_evaluations": values[:, 2].astype(np.int64),
        "k": values[:, 3],
        "loss_total": values[:, 4],
        "loss_physics": values[:, 5],
        "loss_data": values[:, 6],
        "elapsed_seconds": np.asarray(elapsed),
        "total_loss_evaluations": np.asarray(n_evals),
    }


def evaluate(model: InversePINN, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(grid[:, None], requires_grad=True)
    prediction = model(x).detach().numpy().ravel()
    residual = normalized_residual(model, x).detach().numpy().ravel()
    return prediction, residual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adam-steps", type=int, default=4000)
    parser.add_argument("--lbfgs-steps", type=int, default=40)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(SEED)
    np_rng = np.random.default_rng(SEED)

    x_data = np.linspace(0.05, 0.95, SENSOR_COUNT)
    y_clean = np.sin(np.pi * x_data) / TRUE_K
    # Independent additive Gaussian errors, sigma = 1% of exact peak amplitude.
    noise_sigma = 0.01 / TRUE_K
    y_data = y_clean + np_rng.normal(0.0, noise_sigma, size=SENSOR_COUNT)
    x_obs = torch.tensor(x_data[:, None])
    y_obs = torch.tensor(y_data[:, None])
    # Interior collocation points; the hard ansatz enforces both endpoints.
    x_collocation = np.linspace(0.0, 1.0, 130)[1:-1]
    x_r = torch.tensor(x_collocation[:, None], requires_grad=True)

    model_hybrid = InversePINN()
    model_physics_only = copy.deepcopy(model_hybrid)
    histories = {}
    for name, model, weight in (
        ("hybrid", model_hybrid, DATA_WEIGHT),
        ("physics_only", model_physics_only, 0.0),
    ):
        histories[name] = train(model, x_r, x_obs, y_obs, weight,
                                args.adam_steps, args.lbfgs_steps)
        print(f"{name}: k={model.k.item():.6f}, "
              f"time={histories[name]['elapsed_seconds'].item():.1f}s", flush=True)

    grid = np.linspace(0.0, 1.0, 1001)
    exact = np.sin(np.pi * grid) / TRUE_K
    # Held-out midpoint grid is different from training collocation and sensors.
    test_grid = (np.arange(1000) + 0.5) / 1000.0
    test_exact = np.sin(np.pi * test_grid) / TRUE_K
    payload = {
        "x": grid,
        "u_true": exact,
        "x_data": x_data,
        "y_data": y_data,
        "y_clean": y_clean,
        "noise_sigma": np.asarray(noise_sigma),
        "x_collocation": x_collocation,
        "true_k": np.asarray(TRUE_K),
        "initial_k": np.asarray(INITIAL_K),
        "data_weight": np.asarray(DATA_WEIGHT),
        "x_test": test_grid,
    }
    metrics = {}
    for name, model in (("hybrid", model_hybrid),
                        ("physics_only", model_physics_only)):
        predicted, residual = evaluate(model, grid)
        test_prediction, _ = evaluate(model, test_grid)
        inferred = model.k.item()
        metrics[name] = {
            "inferred_k": inferred,
            "k_relative_error_percent": 100.0 * abs(inferred - TRUE_K) / TRUE_K,
            "held_out_relative_l2": float(np.linalg.norm(test_prediction - test_exact)
                                         / np.linalg.norm(test_exact)),
            "normalized_residual_rms": float(np.sqrt(np.mean(residual**2))),
            "final_physics_loss": float(histories[name]["loss_physics"][-1]),
            "final_sensor_mse": float(histories[name]["loss_data"][-1]),
            "elapsed_seconds": float(histories[name]["elapsed_seconds"]),
            "loss_evaluations": int(histories[name]["total_loss_evaluations"]),
            "boundary_values": [float(predicted[0]), float(predicted[-1])],
        }
        payload[f"u_{name}"] = predicted
        payload[f"residual_{name}"] = residual
        payload[f"u_test_{name}"] = test_prediction
        for key, value in histories[name].items():
            payload[f"{name}_{key}"] = value

    # Analytic family: every curve has exactly zero PDE and boundary residual.
    # The observations distinguish k values; the PDE alone cannot do that.
    k_family = np.linspace(0.35, 2.0, 501)
    family_at_sensors = np.sin(np.pi * x_data[None, :]) / k_family[:, None]
    payload["family_k"] = k_family
    payload["family_sensor_mse"] = ((family_at_sensors - y_data[None, :])**2).mean(axis=1)
    payload["family_physics_loss"] = np.zeros_like(k_family)
    family_examples = np.asarray([0.5, TRUE_K, INITIAL_K, 1.7])
    payload["family_example_k"] = family_examples
    payload["family_example_u"] = np.sin(np.pi * grid[None, :]) / family_examples[:, None]

    metadata = {
        "experiment": "1D hybrid inverse diffusion PINN",
        "equation": "-k u''(x) = pi^2 sin(pi x), x in (0,1); u(0)=u(1)=0",
        "true_solution": "u(x) = sin(pi x)/0.7",
        "seed": SEED,
        "device": "cpu",
        "dtype": "float64",
        "threads": 1,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "architecture": "1 -> 32 tanh -> 32 tanh -> 1; u=x(1-x)net(x)",
        "positivity": "k=softplus(raw_k)+1e-6",
        "true_k": TRUE_K,
        "initial_k": INITIAL_K,
        "same_initial_parameters_for_both_models": True,
        "sensor_count": SENSOR_COUNT,
        "noise": "Independent additive Gaussian; sigma=1% of exact peak amplitude",
        "noise_sigma": noise_sigma,
        "collocation_count": len(x_collocation),
        "collocation": "128 fixed equally spaced interior points",
        "loss": "mean((-k*u_xx/pi^2-sin(pi*x))^2) + data_weight*mean((u(x_data)-y_data)^2)",
        "hybrid_data_weight": DATA_WEIGHT,
        "physics_only_data_weight": 0.0,
        "optimizer": {
            "adam_steps": args.adam_steps,
            "adam_learning_rate": 0.001,
            "lbfgs_outer_steps": args.lbfgs_steps,
            "lbfgs_max_inner_iterations": 5,
            "lbfgs_line_search": "strong_wolfe",
        },
        "history_sampling": "initial state, every 20 Adam updates, each L-BFGS outer step",
        "history_phase_codes": {"0": "Adam", "1": "L-BFGS"},
        "held_out_evaluation": "1000 uniform cell midpoints, independent of collocation and sensors",
        "interpretation": (
            "Physics-only inverse training is nonidentifiable: for every k>0, "
            "sin(pi*x)/k solves the same PDE and boundary conditions. The selected "
            "physics-only k is one optimizer outcome, not a uniquely inferred value."
        ),
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "inverse_1d.npz", **payload)
    (args.output_dir / "inverse_1d.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

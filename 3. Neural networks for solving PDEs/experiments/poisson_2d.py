#!/usr/bin/env python3
"""Reproducible CPU PINN for the Dirichlet Poisson problem on the unit square.

Run from any directory: ``python3 experiments/poisson_2d.py``.
No solution labels enter training. The analytic solution is evaluated only after
optimization, on a separate uniform diagnostic grid. Arrays are stored in (y, x)
order; all PDE residuals and losses are normalized by 2*pi**2.
"""

from __future__ import annotations

import argparse
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


class PoissonPINN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy[:, 0:1], xy[:, 1:2]
        # This factor satisfies all four homogeneous Dirichlet edges exactly.
        return x * (1 - x) * y * (1 - y) * self.net(2 * xy - 1)


# BEGIN LAPLACIAN_SNIPPET
def normalized_residual(model, xy):
    """Return (-Delta u_theta - f)/(2*pi**2)."""
    u = model(xy)
    grad_u = torch.autograd.grad(
        u.sum(), xy, create_graph=True)[0]
    u_xx = torch.autograd.grad(
        grad_u[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(
        grad_u[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    x, y = xy[:, 0:1], xy[:, 1:2]
    f = 2 * math.pi**2 * torch.sin(math.pi*x) * torch.sin(math.pi*y)
    return (-u_xx - u_yy - f) / (2 * math.pi**2)
# END LAPLACIAN_SNIPPET


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    model = PoissonPINN(args.width).cpu()
    sampler = torch.quasirandom.SobolEngine(2, scramble=True, seed=args.seed)
    xy_train = sampler.draw(args.collocation, dtype=torch.float64)
    xy_train.requires_grad_(True)
    losses, phases = [], []

    def evaluate_training_loss():
        return normalized_residual(model, xy_train).square().mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for step in range(args.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        xy_train.grad = None
        loss = evaluate_training_loss()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        phases.append(0)
        if step == 0 or (step + 1) % 500 == 0:
            print(f"Adam {step + 1:4d}: normalized residual MSE {loss.item():.6e}",
                  flush=True)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=args.lbfgs_steps,
        max_eval=args.lbfgs_steps * 2, history_size=50,
        tolerance_grad=1e-11, tolerance_change=1e-13,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        xy_train.grad = None
        loss = evaluate_training_loss()
        loss.backward()
        # LBFGS may evaluate its closure several times per accepted step.
        losses.append(loss.item())
        phases.append(1)
        return loss

    optimizer.step(closure)
    final_train_loss = evaluate_training_loss().item()
    losses.append(final_train_loss)
    phases.append(1)
    training_seconds = time.perf_counter() - start

    x = np.linspace(0.0, 1.0, args.grid_size)
    y = np.linspace(0.0, 1.0, args.grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xy_test = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]))
    with torch.no_grad():
        u_pred = model(xy_test).numpy().reshape(xx.shape)
    # Analytic reference is used here for diagnostics, never in training.
    u_true = np.sin(math.pi * xx) * np.sin(math.pi * yy)
    abs_error = np.abs(u_pred - u_true)
    xy_test.requires_grad_(True)
    residual = normalized_residual(model, xy_test).detach().numpy().reshape(xx.shape)
    boundary = np.concatenate([
        u_pred[0, :], u_pred[-1, :], u_pred[:, 0], u_pred[:, -1]
    ])
    metrics = {
        "relative_l2": float(np.linalg.norm(u_pred-u_true) / np.linalg.norm(u_true)),
        "max_abs_error": float(abs_error.max()),
        "boundary_max_abs": float(np.max(np.abs(boundary))),
        "train_normalized_residual_mse": final_train_loss,
        "test_normalized_residual_mse": float(np.mean(residual**2)),
        "test_normalized_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "interior_test_normalized_residual_rms": float(np.sqrt(np.mean(residual[1:-1, 1:-1]**2))),
        "training_seconds": training_seconds,
        "total_seconds": time.perf_counter() - start,
        "loss_evaluations": len(losses),
        "lbfgs_closure_evaluations": sum(p == 1 for p in phases) - 1,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "poisson_2d.npz", x=x, y=y,
        u_true=u_true, u_pred=u_pred, abs_error=abs_error,
        normalized_residual=residual,
        loss_history=np.asarray(losses), loss_phase=np.asarray(phases),
        collocation_xy=xy_train.detach().numpy(),
    )
    metadata = {
        "experiment": "2D Poisson PINN with exact homogeneous Dirichlet enforcement",
        "equation": "-Delta u = 2*pi^2*sin(pi*x)*sin(pi*y) on [0,1]^2",
        "exact_solution": "sin(pi*x)*sin(pi*y); diagnostics only",
        "ansatz": "x*(1-x)*y*(1-y)*MLP(2*[x,y]-1)",
        "architecture": [2, args.width, args.width, args.width, 1],
        "activation": "tanh",
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config": vars(args),
        "sampling": "fixed scrambled Sobol collocation; uniform diagnostic grid",
        "training_labels": "none; forcing and hard boundary condition only",
        "residual_normalization": "r=(-Delta u-f)/(2*pi**2); loss=mean(r**2)",
        "loss_axis": "objective evaluation; 0=Adam before update, 1=LBFGS closure/final",
        "array_order": "(y, x)",
        "metrics": metrics,
        "versions": {"python": sys.version, "torch": torch.__version__,
                     "numpy": np.__version__, "platform": platform.platform()},
        "provenance": {"script": Path(__file__).name,
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    (output / "poisson_2d.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--collocation", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--adam-steps", type=int, default=2000)
    parser.add_argument("--lbfgs-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()

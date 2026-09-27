"""
14_degenerate_nonlocal_control.py
=================================

Numerical null-control experiment for the degenerate nonlocal parabolic
equation in the Part II slides:

    u_t - ell(int_0^1 u dx) (x^gamma u_x)_x + c(t, x) u = h 1_omega,
    c(t, x) = -20 t,
    ell(r) = 1 + 1 / (1 + r**2),
    u(0, x) = 2 sin(2 pi x).

The default final time is T=1. Use --T to change it.

For gamma < 1, the weakly degenerate boundary condition is
    u(t, 0) = u(t, 1) = 0.
For gamma >= 1, the strongly degenerate boundary condition is
    (x^gamma u_x)(t, 0) = 0,   u(t, 1) = 0.

The control h is computed by a fixed-point linearization:
    1. simulate the nonlinear equation for a reference trajectory,
    2. freeze ell(int u) along that trajectory,
    3. compute the minimal Euclidean-energy discrete control that nulls the
       frozen linear terminal state,
    4. simulate the nonlinear equation with Newton solves for each time step,
    5. repeat until the trajectory update is below eps or max_outer is reached.

Outputs:
    degenerate_nonlocal_control_summary.csv
    degenerate_nonlocal_control_profiles.csv
    degenerate_nonlocal_control.png
    degenerate_nonlocal_control_surface.png
    degenerate_nonlocal_control_surface_gamma05.png
    degenerate_nonlocal_control_surface_gamma10.png
    degenerate_nonlocal_control_surface_gamma15.png
    degenerate_nonlocal_control_surface_variants.png
    degenerate_nonlocal_control_surface_fine.png, with --fine-surface
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.linalg import LinAlgWarning
from scipy.optimize import minimize

# Default output folders sit next to this script.
HERE = Path(__file__).resolve().parent


GAMMAS = (0.5, 1.0, 1.5)
DEFAULT_T = 1.0


@dataclass(frozen=True)
class DomainChoice:
    index: int
    omega: tuple[float, float]
    omega_prime: tuple[float, float]


DOMAINS = (
    DomainChoice(1, (0.2, 0.8), (0.21, 0.79)),
    DomainChoice(2, (0.2, 0.8), (0.3, 0.7)),
    DomainChoice(3, (0.4, 0.6), (0.42, 0.58)),
)


@dataclass
class GridData:
    x: np.ndarray
    weights: np.ndarray
    diffusion: np.ndarray
    bc_type: str


@dataclass
class SparseGridData:
    x: np.ndarray
    weights: np.ndarray
    diffusion: sp.csc_matrix
    bc_type: str


@dataclass
class CaseResult:
    gamma: float
    domain: DomainChoice
    bc_type: str
    x: np.ndarray
    weights: np.ndarray
    u0: np.ndarray
    uncontrolled: np.ndarray
    controlled: np.ndarray
    controls: np.ndarray
    control_indices: np.ndarray
    outer_history: list[dict[str, float]]
    linear_terminal_l2: float
    refine_iterations: int
    final_objective: float


def ell(mass: float) -> float:
    return 1.0 + 1.0 / (1.0 + mass * mass)


def initial_profile(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(2.0 * np.pi * x)


def build_grid(gamma: float, nx: int) -> GridData:
    dx = 1.0 / nx
    global_x = np.linspace(0.0, 1.0, nx + 1)

    if gamma < 1.0:
        bc_type = "weak Dirichlet-Dirichlet"
        unknown_global = np.arange(1, nx, dtype=int)
        weights = np.full(nx - 1, dx)
    else:
        bc_type = "strong Neumann-Dirichlet"
        unknown_global = np.arange(0, nx, dtype=int)
        weights = np.full(nx, dx)
        weights[0] = 0.5 * dx

    local_for_global = {int(g): j for j, g in enumerate(unknown_global)}
    m = len(unknown_global)
    diffusion = np.zeros((m, m), dtype=float)

    for row, i in enumerate(unknown_global):
        # Right flux: a_{i+1/2} (u_{i+1} - u_i) / dx.
        a_plus = ((i + 0.5) * dx) ** gamma
        diffusion[row, row] -= a_plus / dx**2
        j_plus = local_for_global.get(int(i + 1))
        if j_plus is not None:
            diffusion[row, j_plus] += a_plus / dx**2

        # Left flux. In the strongly degenerate case the x=0 flux is imposed
        # as zero; otherwise Dirichlet boundary values are already omitted.
        if i > 0:
            a_minus = ((i - 0.5) * dx) ** gamma
            diffusion[row, row] -= a_minus / dx**2
            j_minus = local_for_global.get(int(i - 1))
            if j_minus is not None:
                diffusion[row, j_minus] += a_minus / dx**2

    return GridData(global_x[unknown_global], weights, diffusion, bc_type)


def refined_nodes(nx: int, domain: DomainChoice) -> np.ndarray:
    features = np.array(
        [
            0.0,
            domain.omega[0],
            domain.omega[1],
            domain.omega_prime[0],
            domain.omega_prime[1],
            1.0,
        ],
        dtype=float,
    )
    sample = np.linspace(0.0, 1.0, max(50_000, 100 * nx))
    density = 1.0 + 10.0 * np.exp(-sample / 0.025)
    for point in features[1:-1]:
        density += 5.0 * np.exp(-((sample - point) / 0.012) ** 2)

    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(sample)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    nodes = np.interp(np.linspace(0.0, 1.0, nx + 1), cdf, sample)
    nodes = np.concatenate((nodes, features))
    return np.unique(np.round(nodes, 14))


def build_sparse_refined_grid(gamma: float, nx: int, domain: DomainChoice) -> SparseGridData:
    nodes = refined_nodes(nx, domain)

    if gamma < 1.0:
        bc_type = "weak Dirichlet-Dirichlet"
        unknown_global = np.arange(1, len(nodes) - 1, dtype=int)
    else:
        bc_type = "strong Neumann-Dirichlet"
        unknown_global = np.arange(0, len(nodes) - 1, dtype=int)

    weights = np.empty(len(unknown_global), dtype=float)
    for row, i in enumerate(unknown_global):
        if i == 0:
            weights[row] = 0.5 * (nodes[1] - nodes[0])
        else:
            weights[row] = 0.5 * (nodes[i + 1] - nodes[i - 1])

    local_for_global = {int(g): j for j, g in enumerate(unknown_global)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add_value(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(value)

    for row, i in enumerate(unknown_global):
        volume = weights[row]

        h_plus = nodes[i + 1] - nodes[i]
        a_plus = (0.5 * (nodes[i + 1] + nodes[i])) ** gamma
        coeff_plus = a_plus / (h_plus * volume)
        add_value(row, row, -coeff_plus)
        j_plus = local_for_global.get(int(i + 1))
        if j_plus is not None:
            add_value(row, j_plus, coeff_plus)

        if not (gamma >= 1.0 and i == 0):
            h_minus = nodes[i] - nodes[i - 1]
            a_minus = (0.5 * (nodes[i] + nodes[i - 1])) ** gamma
            coeff_minus = a_minus / (h_minus * volume)
            add_value(row, row, -coeff_minus)
            j_minus = local_for_global.get(int(i - 1))
            if j_minus is not None:
                add_value(row, j_minus, coeff_minus)

    diffusion = sp.csc_matrix(
        (data, (rows, cols)), shape=(len(unknown_global), len(unknown_global))
    )
    return SparseGridData(nodes[unknown_global], weights, diffusion, bc_type)


def build_refined_grid(gamma: float, nx: int, domain: DomainChoice) -> GridData:
    sparse_grid = build_sparse_refined_grid(gamma, nx, domain)
    return GridData(
        sparse_grid.x,
        sparse_grid.weights,
        sparse_grid.diffusion.toarray(),
        sparse_grid.bc_type,
    )


def l2_norm(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * values * values)))


def spacetime_update_norm(
    current: np.ndarray, previous: np.ndarray, weights: np.ndarray, dt: float
) -> float:
    diff = current - previous
    return float(np.sqrt(dt * np.sum((diff * diff) @ weights)))


def make_step_matrix(diffusion: np.ndarray, dt: float, t: float, ell_value: float) -> np.ndarray:
    m = diffusion.shape[0]
    c_value = -20.0 * t
    lhs = np.eye(m) - dt * ell_value * diffusion + dt * c_value * np.eye(m)
    return lhs


def step_residual(
    u: np.ndarray,
    rhs: np.ndarray,
    diffusion: np.ndarray,
    weights: np.ndarray,
    dt: float,
    t: float,
) -> np.ndarray:
    c_value = -20.0 * t
    mass = float(u @ weights)
    return u - rhs - dt * ell(mass) * (diffusion @ u) + dt * c_value * u


def step_jacobian(
    u: np.ndarray,
    diffusion: np.ndarray,
    weights: np.ndarray,
    dt: float,
    t: float,
) -> np.ndarray:
    m = len(u)
    mass = float(u @ weights)
    ell_value = ell(mass)
    ell_prime = -2.0 * mass / (1.0 + mass * mass) ** 2
    c_value = -20.0 * t
    diffusion_u = diffusion @ u
    return (
        np.eye(m)
        - dt * ell_value * diffusion
        - dt * ell_prime * np.outer(diffusion_u, weights)
        + dt * c_value * np.eye(m)
    )


def solve_nonlinear_step_newton(
    diffusion: np.ndarray,
    weights: np.ndarray,
    previous: np.ndarray,
    rhs: np.ndarray,
    dt: float,
    t: float,
    newton_tol: float,
    newton_max: int,
) -> np.ndarray:
    u = previous.copy()
    residual = step_residual(u, rhs, diffusion, weights, dt, t)
    residual_norm = np.linalg.norm(residual, ord=np.inf)
    scale = max(1.0, np.linalg.norm(rhs, ord=np.inf))

    for _ in range(newton_max):
        if residual_norm <= newton_tol * scale:
            break

        jacobian = step_jacobian(u, diffusion, weights, dt, t)
        delta = la.solve(jacobian, -residual, assume_a="gen")

        damping = 1.0
        accepted = False
        for _ in range(10):
            candidate = u + damping * delta
            candidate_residual = step_residual(candidate, rhs, diffusion, weights, dt, t)
            candidate_norm = np.linalg.norm(candidate_residual, ord=np.inf)
            if candidate_norm <= (1.0 - 1.0e-4 * damping) * residual_norm:
                u = candidate
                residual = candidate_residual
                residual_norm = candidate_norm
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            u = u + delta
            residual = step_residual(u, rhs, diffusion, weights, dt, t)
            residual_norm = np.linalg.norm(residual, ord=np.inf)

        if np.linalg.norm(delta, ord=np.inf) <= newton_tol * max(1.0, np.linalg.norm(u, ord=np.inf)):
            break

    return u


def transition_matrices(
    diffusion: np.ndarray,
    weights: np.ndarray,
    reference: np.ndarray,
    final_time: float,
) -> list[np.ndarray]:
    nt = reference.shape[0] - 1
    dt = final_time / nt
    identity = np.eye(diffusion.shape[0])
    transitions: list[np.ndarray] = []

    for n in range(nt):
        mass = float(reference[n + 1] @ weights)
        lhs = make_step_matrix(diffusion, dt, (n + 1) * dt, ell(mass))
        transitions.append(la.solve(lhs, identity, assume_a="gen"))

    return transitions


def simulate_nonlinear(
    diffusion: np.ndarray,
    weights: np.ndarray,
    u0: np.ndarray,
    final_time: float,
    controls: np.ndarray,
    control_indices: np.ndarray,
    newton_tol: float = 1.0e-11,
    newton_max: int = 20,
) -> np.ndarray:
    nt = controls.shape[0]
    dt = final_time / nt
    trajectory = np.zeros((nt + 1, len(u0)), dtype=float)
    trajectory[0] = u0

    state = u0.copy()
    for n in range(nt):
        t_next = (n + 1) * dt
        rhs = state.copy()
        control_vec = np.zeros_like(state)
        control_vec[control_indices] = controls[n]
        rhs += dt * control_vec

        new_state = solve_nonlinear_step_newton(
            diffusion, weights, state, rhs, dt, t_next, newton_tol, newton_max
        )

        state = new_state
        trajectory[n + 1] = state

    return trajectory


def simulate_nonlinear_sparse(
    diffusion: sp.csc_matrix,
    weights: np.ndarray,
    u0: np.ndarray,
    final_time: float,
    controls: np.ndarray,
    control_indices: np.ndarray,
    newton_tol: float = 1.0e-11,
    newton_max: int = 20,
) -> np.ndarray:
    nt = controls.shape[0]
    dt = final_time / nt
    m = len(u0)
    dense_diffusion = diffusion.toarray()
    trajectory = np.zeros((nt + 1, m), dtype=float)
    trajectory[0] = u0

    state = u0.copy()
    for n in range(nt):
        t_next = (n + 1) * dt
        rhs = state.copy()
        control_vec = np.zeros_like(state)
        control_vec[control_indices] = controls[n]
        rhs += dt * control_vec

        new_state = solve_nonlinear_step_newton(
            dense_diffusion, weights, state, rhs, dt, t_next, newton_tol, newton_max
        )

        state = new_state
        trajectory[n + 1] = state

    return trajectory


def simulate_nonlinear_with_jacobians(
    diffusion: np.ndarray,
    weights: np.ndarray,
    u0: np.ndarray,
    final_time: float,
    controls: np.ndarray,
    control_indices: np.ndarray,
    newton_tol: float = 1.0e-11,
    newton_max: int = 20,
) -> tuple[np.ndarray, list[np.ndarray]]:
    nt = controls.shape[0]
    dt = final_time / nt
    m = len(u0)
    trajectory = np.zeros((nt + 1, m), dtype=float)
    jacobians: list[np.ndarray] = []
    trajectory[0] = u0

    state = u0.copy()
    for n in range(nt):
        t_next = (n + 1) * dt
        rhs = state.copy()
        control_vec = np.zeros_like(state)
        control_vec[control_indices] = controls[n]
        rhs += dt * control_vec

        new_state = solve_nonlinear_step_newton(
            diffusion, weights, state, rhs, dt, t_next, newton_tol, newton_max
        )
        jacobian = step_jacobian(new_state, diffusion, weights, dt, t_next)
        jacobians.append(jacobian)

        state = new_state
        trajectory[n + 1] = state

    return trajectory, jacobians


def linear_null_control(
    diffusion: np.ndarray,
    weights: np.ndarray,
    u0: np.ndarray,
    reference: np.ndarray,
    final_time: float,
    control_indices: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, float]:
    transitions = transition_matrices(diffusion, weights, reference, final_time)
    nt = len(transitions)
    dt = final_time / nt
    m = len(u0)

    free_terminal = u0.copy()
    for step_matrix in transitions:
        free_terminal = step_matrix @ free_terminal

    suffix = np.eye(m)
    gramian = regularization * np.eye(m)
    control_maps: list[np.ndarray] = [np.empty((m, len(control_indices)))] * nt

    for n in range(nt - 1, -1, -1):
        control_map = suffix @ (dt * transitions[n][:, control_indices])
        control_maps[n] = control_map
        gramian += control_map @ control_map.T
        suffix = suffix @ transitions[n]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        try:
            multiplier = la.solve(gramian, -free_terminal, assume_a="pos")
        except la.LinAlgError:
            multiplier = la.solve(gramian, -free_terminal, assume_a="gen")

    controls = np.zeros((nt, len(control_indices)), dtype=float)
    terminal = free_terminal.copy()
    for n, control_map in enumerate(control_maps):
        controls[n] = control_map.T @ multiplier
        terminal += control_map @ controls[n]

    return controls, l2_norm(terminal, weights)


def objective_and_gradient(
    flat_controls: np.ndarray,
    diffusion: np.ndarray,
    weights: np.ndarray,
    u0: np.ndarray,
    final_time: float,
    control_indices: np.ndarray,
    control_alpha: float,
) -> tuple[float, np.ndarray]:
    q = len(control_indices)
    nt = flat_controls.size // q
    controls = flat_controls.reshape(nt, q)
    dt = final_time / nt
    control_weights = weights[control_indices]

    trajectory, jacobians = simulate_nonlinear_with_jacobians(
        diffusion, weights, u0, final_time, controls, control_indices
    )
    terminal = trajectory[-1]
    terminal_cost = 0.5 * float(np.sum(weights * terminal * terminal))
    control_cost = 0.5 * control_alpha * dt * float(np.sum((controls * controls) @ control_weights))
    objective = terminal_cost + control_cost

    gradient = np.zeros_like(controls)
    rhs = -(weights * terminal)
    multiplier_next: np.ndarray | None = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        for n in range(nt - 1, -1, -1):
            if multiplier_next is None:
                multiplier = la.solve(jacobians[n].T, rhs, assume_a="gen")
            else:
                multiplier = la.solve(jacobians[n].T, multiplier_next, assume_a="gen")
            gradient[n] = control_alpha * dt * control_weights * controls[n]
            gradient[n] -= dt * multiplier[control_indices]
            multiplier_next = multiplier

    return objective, gradient.ravel()


def refine_control(
    diffusion: np.ndarray,
    weights: np.ndarray,
    u0: np.ndarray,
    final_time: float,
    controls: np.ndarray,
    control_indices: np.ndarray,
    control_alpha: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    if maxiter <= 0:
        trajectory = simulate_nonlinear(diffusion, weights, u0, final_time, controls, control_indices)
        objective, _ = objective_and_gradient(
            controls.ravel(), diffusion, weights, u0, final_time, control_indices, control_alpha
        )
        return controls, trajectory, 0, objective

    initial_objective, _ = objective_and_gradient(
        controls.ravel(), diffusion, weights, u0, final_time, control_indices, control_alpha
    )

    result = minimize(
        fun=lambda z: objective_and_gradient(
            z, diffusion, weights, u0, final_time, control_indices, control_alpha
        ),
        x0=controls.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": maxiter,
            "ftol": 1.0e-18,
            "gtol": 1.0e-10,
            "maxls": 30,
            "maxcor": 20,
        },
    )

    if result.fun <= initial_objective:
        refined_controls = result.x.reshape(controls.shape)
        refined_objective = float(result.fun)
        iterations = int(result.nit)
    else:
        refined_controls = controls
        refined_objective = float(initial_objective)
        iterations = 0

    trajectory = simulate_nonlinear(
        diffusion, weights, u0, final_time, refined_controls, control_indices
    )
    return refined_controls, trajectory, iterations, refined_objective


def control_l2_norm(
    controls: np.ndarray,
    control_indices: np.ndarray,
    weights: np.ndarray,
    final_time: float,
) -> float:
    nt = controls.shape[0]
    dt = final_time / nt
    control_weights = weights[control_indices]
    return float(np.sqrt(dt * np.sum((controls * controls) @ control_weights)))


def solve_case(
    gamma: float,
    domain: DomainChoice,
    nx: int,
    nt: int,
    final_time: float,
    max_outer: int,
    eps: float,
    regularization: float,
    refine_iters: int,
    control_alpha: float,
    mesh_kind: str = "uniform",
) -> CaseResult:
    if mesh_kind == "feature":
        grid = build_refined_grid(gamma, nx, domain)
    else:
        grid = build_grid(gamma, nx)
    u0 = initial_profile(grid.x)
    control_indices = np.flatnonzero(
        (grid.x >= domain.omega[0]) & (grid.x <= domain.omega[1])
    )
    if len(control_indices) == 0:
        raise ValueError(f"No control nodes found in omega={domain.omega}. Increase nx.")

    zero_controls = np.zeros((nt, len(control_indices)), dtype=float)
    uncontrolled = simulate_nonlinear(
        grid.diffusion, grid.weights, u0, final_time, zero_controls, control_indices
    )

    reference = uncontrolled
    controlled = uncontrolled
    controls = zero_controls
    linear_terminal_l2 = l2_norm(uncontrolled[-1], grid.weights)
    history: list[dict[str, float]] = []
    best_terminal_l2 = np.inf
    best_control = controls
    best_controlled = controlled

    for outer in range(1, max_outer + 1):
        controls, linear_terminal_l2 = linear_null_control(
            grid.diffusion,
            grid.weights,
            u0,
            reference,
            final_time,
            control_indices,
            regularization,
        )
        candidate = simulate_nonlinear(
            grid.diffusion, grid.weights, u0, final_time, controls, control_indices
        )
        dt = final_time / nt
        update = spacetime_update_norm(candidate, reference, grid.weights, dt)
        terminal_l2 = l2_norm(candidate[-1], grid.weights)
        cnorm = control_l2_norm(controls, control_indices, grid.weights, final_time)
        history.append(
            {
                "outer": float(outer),
                "update_l2_q": update,
                "terminal_l2": terminal_l2,
                "control_l2_q": cnorm,
                "linear_terminal_l2": linear_terminal_l2,
            }
        )

        controlled = candidate
        if terminal_l2 < best_terminal_l2:
            best_terminal_l2 = terminal_l2
            best_control = controls.copy()
            best_controlled = candidate.copy()
        if update <= eps:
            break
        reference = candidate

    controls = best_control
    controlled = best_controlled
    controls, controlled, refine_iterations, final_objective = refine_control(
        grid.diffusion,
        grid.weights,
        u0,
        final_time,
        controls,
        control_indices,
        control_alpha,
        refine_iters if l2_norm(controlled[-1], grid.weights) > eps else 0,
    )

    return CaseResult(
        gamma=gamma,
        domain=domain,
        bc_type=grid.bc_type,
        x=grid.x,
        weights=grid.weights,
        u0=u0,
        uncontrolled=uncontrolled,
        controlled=controlled,
        controls=controls,
        control_indices=control_indices,
        outer_history=history,
        linear_terminal_l2=linear_terminal_l2,
        refine_iterations=refine_iterations,
        final_objective=final_objective,
    )


def write_summary(results: list[CaseResult], final_time: float, nx: int, nt: int, outdir: Path) -> None:
    summary_path = outdir / "degenerate_nonlocal_control_summary.csv"
    profile_path = outdir / "degenerate_nonlocal_control_profiles.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            "gamma,domain,bc_type,omega_left,omega_right,omega_prime_left,"
            "omega_prime_right,T,nx,nt,outer_iters,uncontrolled_terminal_l2,"
            "controlled_terminal_l2,relative_terminal_l2,control_l2_q,"
            "last_update_l2_q,linear_terminal_l2,refine_iters,final_objective\n"
        )
        for result in results:
            last = result.outer_history[-1] if result.outer_history else {}
            uncontrolled_l2 = l2_norm(result.uncontrolled[-1], result.weights)
            controlled_l2 = l2_norm(result.controlled[-1], result.weights)
            initial_l2 = l2_norm(result.u0, result.weights)
            relative_l2 = controlled_l2 / max(initial_l2, 1.0e-15)
            final_control_l2 = control_l2_norm(
                result.controls, result.control_indices, result.weights, final_time
            )
            f.write(
                f"{result.gamma:.6g},{result.domain.index},{result.bc_type},"
                f"{result.domain.omega[0]:.12g},{result.domain.omega[1]:.12g},"
                f"{result.domain.omega_prime[0]:.12g},{result.domain.omega_prime[1]:.12g},"
                f"{final_time:.12g},{nx},{nt},{len(result.outer_history)},"
                f"{uncontrolled_l2:.16e},{controlled_l2:.16e},{relative_l2:.16e},"
                f"{final_control_l2:.16e},"
                f"{last.get('update_l2_q', np.nan):.16e},"
                f"{result.linear_terminal_l2:.16e},{result.refine_iterations},"
                f"{result.final_objective:.16e}\n"
            )

    with profile_path.open("w", encoding="utf-8") as f:
        f.write("gamma,domain,x,u0,uncontrolled_T,controlled_T\n")
        for result in results:
            for x, u0, u_un, u_ctl in zip(
                result.x, result.u0, result.uncontrolled[-1], result.controlled[-1]
            ):
                f.write(
                    f"{result.gamma:.6g},{result.domain.index},{x:.16e},"
                    f"{u0:.16e},{u_un:.16e},{u_ctl:.16e}\n"
                )


def make_plot(results: list[CaseResult], final_time: float, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.5), sharex=True, constrained_layout=True)
    result_map = {(r.gamma, r.domain.index): r for r in results}
    legend_axis = None

    for row, gamma in enumerate(GAMMAS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]
            result = result_map.get((gamma, domain.index))
            if result is None:
                ax.axis("off")
                continue

            ax.axvspan(
                domain.omega[0],
                domain.omega[1],
                color="#d9ead3",
                alpha=0.55,
                label=r"$\omega$" if row == 0 and col == 0 else None,
            )
            ax.axvspan(
                domain.omega_prime[0],
                domain.omega_prime[1],
                color="#fce5cd",
                alpha=0.45,
                label=r"$\omega'$" if row == 0 and col == 0 else None,
            )
            ax.plot(result.x, result.u0, color="#757575", linewidth=1.6, label="initial")
            ax.plot(
                result.x,
                result.uncontrolled[-1],
                color="#c0392b",
                linewidth=1.6,
                label="uncontrolled T",
            )
            ax.plot(
                result.x,
                result.controlled[-1],
                color="#1565c0",
                linewidth=1.8,
                label="controlled T",
            )
            ax.axhline(0.0, color="#111111", linewidth=0.7, alpha=0.45)
            ax.set_title(rf"$\gamma={gamma}$, domain {domain.index}", fontsize=11)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel("u")
            if row == 2:
                ax.set_xlabel("x")
            if legend_axis is None:
                legend_axis = ax

    if legend_axis is not None:
        legend_axis.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        rf"Degenerate nonlocal PDE null control, $T={final_time:g}$",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(outdir / "degenerate_nonlocal_control.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def trajectory_with_boundaries(result: CaseResult, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nt = trajectory.shape[0]
    if result.gamma < 1.0:
        x_vals = np.concatenate(([0.0], result.x, [1.0]))
        values = np.zeros((nt, len(x_vals)), dtype=float)
        values[:, 1:-1] = trajectory
    else:
        x_vals = np.concatenate((result.x, [1.0]))
        values = np.zeros((nt, len(x_vals)), dtype=float)
        values[:, :-1] = trajectory
    return x_vals, values


def surface_grid(
    result: CaseResult, trajectory: np.ndarray, final_time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals, values = trajectory_with_boundaries(result, trajectory)
    time_vals = np.linspace(0.0, final_time, values.shape[0])
    x_grid, t_grid = np.meshgrid(x_vals, time_vals)
    return x_grid, t_grid, values


def gamma_filename_tag(gamma: float) -> str:
    return f"{int(round(10.0 * gamma)):02d}"


def plot_solution_3d(
    ax,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    values: np.ndarray,
    *,
    style: str,
    elev: float = 24.0,
    azim: float = -135.0,
) -> None:
    surface_stride_t = max(1, values.shape[0] // 120)
    surface_stride_x = max(1, values.shape[1] // 120)
    mesh_stride_t = max(1, values.shape[0] // 26)
    mesh_stride_x = max(1, values.shape[1] // 34)
    z_min = float(np.nanmin(values))
    z_max = float(np.nanmax(values))
    z_span = max(z_max - z_min, 1.0e-12)

    if style == "surface_mesh":
        ax.plot_surface(
            x_grid,
            t_grid,
            values,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            alpha=0.96,
            rstride=surface_stride_t,
            cstride=surface_stride_x,
        )
        ax.plot_wireframe(
            x_grid,
            t_grid,
            values,
            rstride=mesh_stride_t,
            cstride=mesh_stride_x,
            color="#172033",
            linewidth=0.28,
            alpha=0.5,
        )
    elif style == "wireframe":
        ax.plot_wireframe(
            x_grid,
            t_grid,
            values,
            rstride=max(1, values.shape[0] // 38),
            cstride=max(1, values.shape[1] // 46),
            color="#0b3d91",
            linewidth=0.45,
            alpha=0.9,
        )
    elif style == "contour":
        floor = z_min - 0.08 * z_span
        ax.plot_surface(
            x_grid,
            t_grid,
            values,
            cmap="viridis",
            linewidth=0.0,
            antialiased=True,
            alpha=0.82,
            rstride=surface_stride_t,
            cstride=surface_stride_x,
        )
        ax.plot_wireframe(
            x_grid,
            t_grid,
            values,
            rstride=mesh_stride_t,
            cstride=mesh_stride_x,
            color="#172033",
            linewidth=0.18,
            alpha=0.28,
        )
        ax.contour(
            x_grid,
            t_grid,
            values,
            zdir="z",
            offset=floor,
            levels=14,
            cmap="magma",
            linewidths=0.65,
        )
        ax.set_zlim(floor, z_max)
    else:
        raise ValueError(f"Unknown 3D plot style: {style}")

    ax.set_xlabel("x", labelpad=2)
    ax.set_ylabel("t", labelpad=2)
    ax.set_zlabel("u", labelpad=2)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.tick_params(axis="z", labelsize=7, pad=1)
    ax.set_box_aspect((1.35, 1.0, 0.72), zoom=1.28)
    ax.view_init(elev=elev, azim=azim)


def make_surface_plot(results: list[CaseResult], final_time: float, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_results = len(results)
    if n_results == 0:
        return

    if n_results == 1:
        rows, cols = 1, 1
    elif n_results <= 3:
        rows, cols = 1, n_results
    else:
        cols = 3
        rows = int(np.ceil(n_results / cols))

    fig = plt.figure(figsize=(5.2 * cols, 4.25 * rows), constrained_layout=True)

    for i, result in enumerate(results, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        x_grid, t_grid, values = surface_grid(result, result.controlled, final_time)
        plot_solution_3d(ax, x_grid, t_grid, values, style="surface_mesh")
        ax.set_title(rf"$\gamma={result.gamma:g}$, domain {result.domain.index}", fontsize=10)

    fig.suptitle(
        rf"Controlled solution surface with visible mesh, $T={final_time:g}$",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(outdir / "degenerate_nonlocal_control_surface.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def make_surface_row_plots(
    results: list[CaseResult], final_time: float, outdir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    saved: list[str] = []
    for gamma in sorted({result.gamma for result in results}):
        gamma_results = sorted(
            [result for result in results if result.gamma == gamma],
            key=lambda result: result.domain.index,
        )
        if not gamma_results:
            continue

        fig = plt.figure(figsize=(22.5, 7.2), constrained_layout=False)
        fig.subplots_adjust(left=0.025, right=0.992, bottom=0.055, top=0.84, wspace=0.42)
        for col, result in enumerate(gamma_results, start=1):
            ax = fig.add_subplot(1, len(gamma_results), col, projection="3d")
            x_grid, t_grid, values = surface_grid(result, result.controlled, final_time)
            plot_solution_3d(ax, x_grid, t_grid, values, style="surface_mesh")
            ax.set_title(
                rf"Domain {result.domain.index}: $\omega={result.domain.omega}$",
                fontsize=24,
                fontweight="bold",
                pad=20,
            )

        filename = f"degenerate_nonlocal_control_surface_gamma{gamma_filename_tag(gamma)}.png"
        fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
        plt.close(fig)
        saved.append(filename)

    return saved


def make_surface_variant_plot(
    results: list[CaseResult], final_time: float, outdir: Path
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if not results:
        return None

    result = max(results, key=lambda item: (item.gamma, item.domain.index))
    x_grid, t_grid, values = surface_grid(result, result.controlled, final_time)
    variants = (
        ("surface_mesh", "Surface with mesh overlay"),
        ("wireframe", "Wireframe mesh"),
        ("contour", "Surface with contour projection"),
    )

    fig = plt.figure(figsize=(22.5, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.992, bottom=0.055, top=0.84, wspace=0.42)
    for i, (style, title) in enumerate(variants, start=1):
        ax = fig.add_subplot(1, len(variants), i, projection="3d")
        plot_solution_3d(ax, x_grid, t_grid, values, style=style)
        ax.set_title(title, fontsize=24, fontweight="bold", pad=20)

    filename = "degenerate_nonlocal_control_surface_variants.png"
    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return filename


def interpolate_controls_to_grid(
    result: CaseResult,
    fine_x: np.ndarray,
    fine_control_indices: np.ndarray,
    final_time: float,
    nt_fine: int,
) -> np.ndarray:
    fine_controls = np.zeros((nt_fine, len(fine_control_indices)), dtype=float)
    if len(fine_control_indices) == 0:
        return fine_controls

    coarse_time = np.linspace(
        final_time / result.controls.shape[0],
        final_time,
        result.controls.shape[0],
    )
    fine_time = np.linspace(final_time / nt_fine, final_time, nt_fine)
    coarse_x = result.x[result.control_indices]
    fine_control_x = fine_x[fine_control_indices]
    omega_left, omega_right = result.domain.omega

    for n, time in enumerate(fine_time):
        upper = int(np.searchsorted(coarse_time, time, side="left"))
        if upper <= 0:
            coarse_values = result.controls[0]
        elif upper >= len(coarse_time):
            coarse_values = result.controls[-1]
        else:
            lower = upper - 1
            theta = (time - coarse_time[lower]) / (coarse_time[upper] - coarse_time[lower])
            coarse_values = (1.0 - theta) * result.controls[lower] + theta * result.controls[upper]

        x_ext = np.concatenate(([omega_left], coarse_x, [omega_right]))
        values_ext = np.concatenate(([0.0], coarse_values, [0.0]))
        fine_controls[n] = np.interp(fine_control_x, x_ext, values_ext, left=0.0, right=0.0)

    return fine_controls


def make_fine_surface_plot(
    results: list[CaseResult],
    final_time: float,
    outdir: Path,
    fine_nx: int,
    fine_nt: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_results = len(results)
    if n_results == 0:
        return

    if n_results == 1:
        rows, cols = 1, 1
    elif n_results <= 3:
        rows, cols = 1, n_results
    else:
        cols = 3
        rows = int(np.ceil(n_results / cols))

    fig = plt.figure(figsize=(5.2 * cols, 4.25 * rows), constrained_layout=True)

    for i, result in enumerate(results, start=1):
        fine_grid = build_sparse_refined_grid(result.gamma, fine_nx, result.domain)
        fine_u0 = initial_profile(fine_grid.x)
        fine_control_indices = np.flatnonzero(
            (fine_grid.x >= result.domain.omega[0]) & (fine_grid.x <= result.domain.omega[1])
        )
        fine_controls = interpolate_controls_to_grid(
            result, fine_grid.x, fine_control_indices, final_time, fine_nt
        )
        fine_trajectory = simulate_nonlinear_sparse(
            fine_grid.diffusion,
            fine_grid.weights,
            fine_u0,
            final_time,
            fine_controls,
            fine_control_indices,
        )

        fine_result = CaseResult(
            gamma=result.gamma,
            domain=result.domain,
            bc_type=fine_grid.bc_type,
            x=fine_grid.x,
            weights=fine_grid.weights,
            u0=fine_u0,
            uncontrolled=fine_trajectory,
            controlled=fine_trajectory,
            controls=fine_controls,
            control_indices=fine_control_indices,
            outer_history=[],
            linear_terminal_l2=np.nan,
            refine_iterations=0,
            final_objective=np.nan,
        )
        x_grid, t_grid, values = surface_grid(fine_result, fine_trajectory, final_time)

        ax = fig.add_subplot(rows, cols, i, projection="3d")
        plot_solution_3d(ax, x_grid, t_grid, values, style="surface_mesh")
        ax.set_title(rf"$\gamma={result.gamma:g}$, domain {result.domain.index}", fontsize=10)

    fig.suptitle(
        rf"Feature-refined controlled solution surface with visible mesh, $T={final_time:g}$",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(
        outdir / "degenerate_nonlocal_control_surface_fine.png",
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=float, default=DEFAULT_T, help="final time")
    parser.add_argument("--nx", type=int, default=60, help="number of intervals in x")
    parser.add_argument("--nt", type=int, default=120, help="number of time steps")
    parser.add_argument(
        "--mesh",
        choices=("uniform", "feature"),
        default="uniform",
        help="spatial mesh: uniform or feature-refined near x=0 and omega interfaces",
    )
    parser.add_argument("--max-outer", type=int, default=8, help="fixed-point iterations")
    parser.add_argument("--eps", type=float, default=1.0e-7, help="trajectory update tolerance")
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-12,
        help="Gramian regularization for the discrete null control",
    )
    parser.add_argument(
        "--refine-iters",
        type=int,
        default=50,
        help="L-BFGS iterations for nonlinear adjoint refinement",
    )
    parser.add_argument(
        "--control-alpha",
        type=float,
        default=1.0e-12,
        help="control penalty in the nonlinear refinement objective",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        choices=GAMMAS,
        default=None,
        help="run a single gamma value",
    )
    parser.add_argument(
        "--domain",
        type=int,
        choices=[domain.index for domain in DOMAINS],
        default=None,
        help="run a single domain choice",
    )
    parser.add_argument("--outdir", type=Path, default=HERE / "figures", help="figure directory")
    parser.add_argument("--datadir", type=Path, default=HERE / "results", help="CSV table directory")
    parser.add_argument(
        "--fine-surface",
        action="store_true",
        help="also render a feature-refined high-resolution 3D surface",
    )
    parser.add_argument(
        "--fine-nx",
        type=int,
        default=480,
        help="target intervals for the feature-refined fine surface grid",
    )
    parser.add_argument(
        "--fine-nt",
        type=int,
        default=800,
        help="time steps for the feature-refined fine surface solve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.datadir.mkdir(parents=True, exist_ok=True)

    gammas = [args.gamma] if args.gamma is not None else list(GAMMAS)
    domains = [d for d in DOMAINS if args.domain in (None, d.index)]

    results: list[CaseResult] = []
    for gamma in gammas:
        for domain in domains:
            print(
                f"Solving gamma={gamma:g}, domain={domain.index}, "
                f"omega={domain.omega}, omega'={domain.omega_prime}"
            )
            result = solve_case(
                gamma=gamma,
                domain=domain,
                nx=args.nx,
                nt=args.nt,
                final_time=args.T,
                max_outer=args.max_outer,
                eps=args.eps,
                regularization=args.regularization,
                refine_iters=args.refine_iters,
                control_alpha=args.control_alpha,
                mesh_kind=args.mesh,
            )
            results.append(result)
            last = result.outer_history[-1] if result.outer_history else {}
            print(
                "  "
                f"outer={len(result.outer_history)}, "
                f"||u(T)||_L2={l2_norm(result.controlled[-1], result.weights):.3e}, "
                f"||h||_L2(Qomega)="
                f"{control_l2_norm(result.controls, result.control_indices, result.weights, args.T):.3e}, "
                f"update={last.get('update_l2_q', np.nan):.3e}, "
                f"refine={result.refine_iterations}"
            )

    write_summary(results, args.T, args.nx, args.nt, args.datadir)
    make_plot(results, args.T, args.outdir)
    make_surface_plot(results, args.T, args.outdir)
    row_surface_files = make_surface_row_plots(results, args.T, args.outdir)
    variant_file = make_surface_variant_plot(results, args.T, args.outdir)
    if args.fine_surface:
        make_fine_surface_plot(results, args.T, args.outdir, args.fine_nx, args.fine_nt)
    print("Saved degenerate_nonlocal_control_summary.csv")
    print("Saved degenerate_nonlocal_control_profiles.csv")
    print("Saved degenerate_nonlocal_control.png")
    print("Saved degenerate_nonlocal_control_surface.png")
    for filename in row_surface_files:
        print(f"Saved {filename}")
    if variant_file is not None:
        print(f"Saved {variant_file}")
    if args.fine_surface:
        print("Saved degenerate_nonlocal_control_surface_fine.png")


if __name__ == "__main__":
    main()

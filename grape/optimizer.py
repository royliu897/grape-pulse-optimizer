"""
GRAPE optimizer module.

Optimizers
----------
grape_lbfgs            -- L-BFGS-B (always the inner solver)
grape_multistart       -- N parallel L-BFGS-B runs, return best
grape_basinhopping     -- scipy basin hopping wrapping L-BFGS-B
grape_gradient_descent -- simple GD (debug/learning only)

All accept regularization kwargs (alpha, beta, gamma) that are passed
to the regularization module to produce hardware-friendly smooth pulses.

Flags (exposed in run_grape.py CLI)
------------------------------------
--multistart N     run N parallel L-BFGS-B starts, keep best
--basinhopping     wrap L-BFGS-B in scipy basin hopping
Both can be combined: basin hopping with N random restarts.

Regularization parameters (set in gate YAML or CLI)
----------------------------------------------------
alpha  : power penalty        (default 0.0)
beta   : smoothness penalty   (default 0.01)  <-- primary hardware fix
gamma  : Gaussian envelope    (default 0.0)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize, basinhopping

from physics.propagator import pulse_to_H_list, compute_propagators
from grape.fidelity import gate_fidelity, infidelity
from grape.gradients import compute_gradients
from grape.regularization import regularization_loss


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GRAPEResult:
    Omega_opt        : np.ndarray
    phi_opt          : np.ndarray
    fidelity         : float
    infidelity       : float
    fidelity_history : List[float] = field(default_factory=list)
    n_iterations     : int         = 0
    converged        : bool        = False
    runtime_s        : float       = 0.0
    message          : str         = ""
    reg_breakdown    : dict        = field(default_factory=dict)

    def summary(self):
        print(f"Fidelity:    {self.fidelity:.8f}")
        print(f"Infidelity:  {self.infidelity:.2e}")
        print(f"Iterations:  {self.n_iterations}")
        print(f"Converged:   {self.converged}")
        print(f"Runtime:     {self.runtime_s:.2f}s")
        print(f"Message:     {self.message}")
        if self.reg_breakdown:
            print(f"Reg terms:   {self.reg_breakdown}")


# ---------------------------------------------------------------------------
# Pulse initialization
# ---------------------------------------------------------------------------

def initialize_pulse(N, Omega_mean=1.6, Omega_std=0.1,
                     phi_mean=0.0, phi_std=0.3, seed=None):
    rng        = np.random.default_rng(seed)
    Omega_list = np.clip(Omega_mean + rng.normal(0, Omega_std, N), 0.1, None)
    phi_list   = phi_mean + rng.normal(0, phi_std, N)
    return Omega_list, phi_list


def initialize_pulse_flat(N, Omega=1.6, phi=0.0):
    return np.ones(N) * Omega, np.ones(N) * phi


# ---------------------------------------------------------------------------
# Shared objective builder
# ---------------------------------------------------------------------------

def _make_objective(build_H_func, dt, U_target, qubit_indices, H_kwargs,
                    N, alpha, beta, gamma, fidelity_history, verbose,
                    verbose_every=50):
    """
    Build the (value, gradient) objective function for scipy.minimize.

    Combines fidelity infidelity + regularization into a single scalar.
    """
    def objective(x):
        Omega, phi = x[:N], x[N:]

        # fidelity gradient
        grad_O, grad_p, F_val = compute_gradients(
            Omega, phi, build_H_func, dt,
            U_target, qubit_indices, H_kwargs
        )
        L_fid   = 1.0 - F_val
        g_fid   = np.concatenate([grad_O, grad_p])

        # regularization
        L_reg, rg_O, rg_p, breakdown = regularization_loss(
            Omega, phi, dt, alpha=alpha, beta=beta, gamma=gamma)
        g_reg   = np.concatenate([rg_O, rg_p])

        L_total = L_fid + L_reg
        g_total = g_fid + g_reg

        fidelity_history.append(F_val)

        if verbose and len(fidelity_history) % verbose_every == 0:
            print(f"  iter {len(fidelity_history):4d}  |  "
                  f"F={F_val:.6f}  L_fid={L_fid:.2e}  L_reg={L_reg:.2e}")

        return L_total, g_total

    return objective


# ---------------------------------------------------------------------------
# Optimizer 1: L-BFGS-B (primary solver)
# ---------------------------------------------------------------------------

def grape_lbfgs(Omega_init, phi_init, build_H_func, dt,
                U_target, qubit_indices, H_kwargs=None,
                max_iter=1000, tol=1e-10, verbose=True,
                alpha=0.0, beta=0.01, gamma=0.0):
    """
    GRAPE with L-BFGS-B optimizer + regularization.

    Parameters
    ----------
    Omega_init, phi_init : arrays, length N
    build_H_func : callable
    dt : float
    U_target : array
    qubit_indices : list
    H_kwargs : dict, optional
    max_iter : int
    tol : float
    verbose : bool
    alpha : float
        Power penalty weight. Penalizes sum(Omega^2)*dt.
    beta : float
        Smoothness penalty weight. Penalizes sum((dOmega/dt)^2 + (dphi/dt)^2)*dt.
        This is the primary regularizer for hardware compatibility.
        Suggested: beta = 0.01 for 10MHz AOM bandwidth.
    gamma : float
        Gaussian envelope penalty. 0 = off.

    Returns
    -------
    GRAPEResult
    """
    if H_kwargs is None:
        H_kwargs = {}

    N               = len(Omega_init)
    fidelity_history = []
    t_start         = time.time()

    objective = _make_objective(
        build_H_func, dt, U_target, qubit_indices, H_kwargs,
        N, alpha, beta, gamma, fidelity_history, verbose)

    bounds = [(1e-6, None)] * N + [(None, None)] * N
    x0     = np.concatenate([Omega_init, phi_init])

    result = minimize(
        objective, x0,
        method  = 'L-BFGS-B',
        jac     = True,
        bounds  = bounds,
        options = {'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
    )

    Omega_opt, phi_opt = result.x[:N], result.x[N:]
    F_final = 1.0 - result.fun   # approximate: fun includes reg terms

    # recompute true fidelity without regularization for reporting
    from physics.propagator import pulse_to_H_list, compute_propagators
    from grape.fidelity import gate_fidelity
    H_list  = pulse_to_H_list(Omega_opt, phi_opt, build_H_func, **H_kwargs)
    _, _, U = compute_propagators(H_list, dt)
    F_true  = gate_fidelity(U, U_target, qubit_indices)

    # get final reg breakdown
    _, _, _, breakdown = regularization_loss(
        Omega_opt, phi_opt, dt, alpha=alpha, beta=beta, gamma=gamma)

    return GRAPEResult(
        Omega_opt        = Omega_opt,
        phi_opt          = phi_opt,
        fidelity         = F_true,
        infidelity       = 1.0 - F_true,
        fidelity_history = fidelity_history,
        n_iterations     = result.nit,
        converged        = result.success,
        runtime_s        = time.time() - t_start,
        message          = result.message,
        reg_breakdown    = breakdown,
    )


# ---------------------------------------------------------------------------
# Optimizer 2: Basin hopping (L-BFGS-B as local solver)
# ---------------------------------------------------------------------------

def grape_basinhopping(Omega_init, phi_init, build_H_func, dt,
                       U_target, qubit_indices, H_kwargs=None,
                       n_iter_bh=50, T=0.1, stepsize=0.3,
                       max_iter_local=500, tol=1e-9, verbose=True,
                       alpha=0.0, beta=0.01, gamma=0.0):
    """
    GRAPE with scipy basin hopping (L-BFGS-B as local minimizer).

    Basin hopping alternates between:
        1. Local L-BFGS-B minimization
        2. Random perturbation of the current best solution
        3. Metropolis acceptance criterion (accept worse solutions
           with probability exp(-delta_L / T))

    This helps escape local minima that pure L-BFGS-B gets stuck in.
    Effective when the fidelity landscape has many similar-depth basins.

    Parameters
    ----------
    n_iter_bh : int
        Number of basin hopping steps (outer loop).
        Each step runs a full L-BFGS-B minimization.
    T : float
        "Temperature" for Metropolis criterion. Higher = more exploration.
        0.1 means we accept solutions up to 0.1 worse than current best.
    stepsize : float
        RMS size of random perturbation at each hop.
    max_iter_local : int
        Max L-BFGS-B iterations per local minimization.
    alpha, beta, gamma : float
        Regularization weights (same as grape_lbfgs).

    Returns
    -------
    GRAPEResult
    """
    if H_kwargs is None:
        H_kwargs = {}

    N                = len(Omega_init)
    fidelity_history = []
    t_start          = time.time()

    objective = _make_objective(
        build_H_func, dt, U_target, qubit_indices, H_kwargs,
        N, alpha, beta, gamma, fidelity_history, verbose,
        verbose_every=100)

    bounds      = [(1e-6, None)] * N + [(None, None)] * N
    x0          = np.concatenate([Omega_init, phi_init])
    minimizer   = {'method': 'L-BFGS-B', 'jac': True, 'bounds': bounds,
                   'options': {'maxiter': max_iter_local, 'ftol': tol, 'gtol': tol}}

    # custom step function that respects Omega > 0 bounds
    class BoundedStep:
        def __init__(self, stepsize, N):
            self.stepsize = stepsize
            self.N        = N
        def __call__(self, x):
            x_new      = x + np.random.randn(len(x)) * self.stepsize
            x_new[:N]  = np.clip(x_new[:N], 1e-6, None)
            return x_new

    result = basinhopping(
        objective, x0,
        minimizer_kwargs = minimizer,
        niter            = n_iter_bh,
        T                = T,
        take_step        = BoundedStep(stepsize, N),
        seed             = 0,
    )

    Omega_opt, phi_opt = result.x[:N], result.x[N:]

    # true fidelity without regularization
    H_list  = pulse_to_H_list(Omega_opt, phi_opt, build_H_func, **H_kwargs)
    _, _, U = compute_propagators(H_list, dt)
    F_true  = gate_fidelity(U, U_target, qubit_indices)

    _, _, _, breakdown = regularization_loss(
        Omega_opt, phi_opt, dt, alpha=alpha, beta=beta, gamma=gamma)

    if verbose:
        print(f"  Basin hopping done: F={F_true:.6f}  "
              f"steps={result.nit}  fun_calls={result.nfev}")

    return GRAPEResult(
        Omega_opt        = Omega_opt,
        phi_opt          = phi_opt,
        fidelity         = F_true,
        infidelity       = 1.0 - F_true,
        fidelity_history = fidelity_history,
        n_iterations     = result.nit,
        converged        = result.lowest_optimization_result.success,
        runtime_s        = time.time() - t_start,
        message          = f"Basin hopping: {result.message}",
        reg_breakdown    = breakdown,
    )


# ---------------------------------------------------------------------------
# Optimizer 3: Multi-start (parallel L-BFGS-B runs)
# ---------------------------------------------------------------------------

def _single_run(args):
    """Worker for one L-BFGS-B run. Top-level for multiprocessing."""
    (run_id, Omega_init, phi_init, build_H_func, dt,
     U_target, qubit_indices, H_kwargs, opt_kwargs) = args

    result = grape_lbfgs(
        Omega_init, phi_init, build_H_func, dt,
        U_target, qubit_indices, H_kwargs,
        verbose=False, **opt_kwargs
    )
    return run_id, result


def grape_multistart(build_H_func, dt, U_target, qubit_indices,
                     H_kwargs=None, N_timesteps=20, n_starts=8,
                     opt_kwargs=None, Omega_mean=1.6, Omega_std=0.2,
                     n_workers=None, verbose=True):
    """
    Multi-start GRAPE: N parallel L-BFGS-B runs, return best.

    Always uses L-BFGS-B as the inner solver (fastest and most reliable).
    Regularization kwargs (alpha, beta, gamma) are passed via opt_kwargs.

    Parameters
    ----------
    build_H_func, dt, U_target, qubit_indices, H_kwargs : see grape_lbfgs
    N_timesteps : int
    n_starts : int
        Number of independent random starts.
    opt_kwargs : dict
        Extra kwargs for grape_lbfgs: max_iter, tol, alpha, beta, gamma.
    Omega_mean, Omega_std : float
        Random initialization parameters.
    n_workers : int, optional
        Parallel workers. Defaults to min(n_starts, cpu_count).
    verbose : bool

    Returns
    -------
    best_result : GRAPEResult
    all_results : list of GRAPEResult, sorted best-first
    """
    if H_kwargs   is None: H_kwargs   = {}
    if opt_kwargs is None: opt_kwargs = {}

    t_start  = time.time()
    run_args = []

    for i in range(n_starts):
        Omega_init, phi_init = initialize_pulse(
            N_timesteps,
            Omega_mean=Omega_mean, Omega_std=Omega_std, seed=i)
        run_args.append((
            i, Omega_init, phi_init, build_H_func, dt,
            U_target, qubit_indices, H_kwargs, opt_kwargs
        ))

    if verbose:
        print(f"  Multi-start: {n_starts} parallel L-BFGS-B runs ...")

    results_dict = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_single_run, args): args[0]
                   for args in run_args}
        for future in as_completed(futures):
            run_id, result = future.result()
            results_dict[run_id] = result
            if verbose:
                print(f"  run {run_id:2d}  F={result.fidelity:.6f}  "
                      f"{'OK' if result.converged else '--'}")

    all_results = sorted(
        [results_dict[i] for i in range(n_starts)],
        key=lambda r: r.fidelity, reverse=True)

    best = all_results[0]
    if verbose:
        print(f"  Best fidelity: {best.fidelity:.8f}  "
              f"({time.time()-t_start:.1f}s)")

    return best, all_results


# ---------------------------------------------------------------------------
# Optimizer 4: simple gradient descent (debug/learning only)
# ---------------------------------------------------------------------------

def grape_gradient_descent(Omega_init, phi_init, build_H_func, dt,
                            U_target, qubit_indices, H_kwargs=None,
                            learning_rate=0.01, max_iter=500,
                            tol=1e-6, verbose=True,
                            alpha=0.0, beta=0.01, gamma=0.0):
    if H_kwargs is None:
        H_kwargs = {}

    Omega            = Omega_init.copy()
    phi              = phi_init.copy()
    fidelity_history = []
    t_start          = time.time()

    for i in range(max_iter):
        grad_O, grad_p, F_val = compute_gradients(
            Omega, phi, build_H_func, dt,
            U_target, qubit_indices, H_kwargs)

        L_reg, rg_O, rg_p, _ = regularization_loss(
            Omega, phi, dt, alpha=alpha, beta=beta, gamma=gamma)

        fidelity_history.append(F_val)
        L = 1.0 - F_val + L_reg

        if verbose and i % 50 == 0:
            print(f"  iter {i:4d}  F={F_val:.6f}  L={L:.2e}")

        if (1.0 - F_val) < tol:
            return GRAPEResult(
                Omega_opt=Omega, phi_opt=phi,
                fidelity=F_val, infidelity=1-F_val,
                fidelity_history=fidelity_history,
                n_iterations=i, converged=True,
                runtime_s=time.time()-t_start,
                message=f"Converged at {i}")

        Omega = np.clip(Omega - learning_rate*(grad_O + rg_O), 1e-6, None)
        phi   = phi   - learning_rate*(grad_p + rg_p)

    F_final = fidelity_history[-1] if fidelity_history else 0.0
    return GRAPEResult(
        Omega_opt=Omega, phi_opt=phi,
        fidelity=F_final, infidelity=1-F_final,
        fidelity_history=fidelity_history,
        n_iterations=max_iter, converged=False,
        runtime_s=time.time()-t_start,
        message=f"Max iterations reached")

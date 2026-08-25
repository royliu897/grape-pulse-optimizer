"""
Main entry point for GRAPE pulse optimization.

Usage
-----
    python run_grape.py --lab labs/real.yaml --gate gates/cz.yaml
    python run_grape.py --lab labs/paper.yaml --gate gates/identity.yaml
    python run_grape.py --lab labs/real.yaml --gate gates/cz.yaml --no-hardware
    python run_grape.py --lab labs/real.yaml --gate gates/gate.yaml --output results/my_run.json

Arguments
---------
    --lab       Path to lab config YAML (e.g. labs/paper.yaml)
    --gate      Path to gate config YAML (e.g. gates/cz.yaml)
    --output    Path to save results JSON (default: results/<lab>_<gate>_<timestamp>.json)
    --no-hardware   Skip hardware correction step (pure GRAPE only)
    --verbose   Print progress during optimization

Output
------
    JSON file containing:
        - lab and gate config used
        - optimized pulse (Omega_list, phi_list)
        - achieved fidelity
        - fidelity history
        - hardware-corrected pulse (if enabled)
        - runtime
        - timestamp
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import yaml
from datetime import datetime

# make sure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics.systems.rydberg import build_H
from physics.systems.two_qubit_rydberg import build_H_two_qubit
from grape.fidelity import (target_X, target_CZ, target_Hadamard,
                             target_Y, target_Z)
from grape.optimizer import (grape_lbfgs, grape_gradient_descent,
                              grape_multistart, initialize_pulse,
                              GRAPEResult)
from grape.two_qubit_gradients import compute_two_qubit_gradients
from hardware.pipeline import Pulse, HardwarePipeline
from hardware.components import AOMModel, FiberModel


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_target_gate(gate_cfg):
    """Return (U_target, qubit_indices) from gate config."""
    name = gate_cfg['target'].upper()
    targets = {
        'IDENTITY': np.eye(2, dtype=complex),
        'X':        target_X(),
        'Y':        target_Y(),
        'Z':        target_Z(),
        'HADAMARD': target_Hadamard(),
        'CZ':       target_CZ(),
    }
    if name not in targets:
        raise ValueError(f"Unknown target gate '{name}'. "
                         f"Options: {list(targets.keys())}")
    return targets[name], gate_cfg.get('qubit_indices', [0, 1])


def build_hardware_pipeline(lab_cfg, seed=42):
    """Build HardwarePipeline from lab config hardware section."""
    hw = lab_cfg.get('hardware', {})
    components = []

    if 'aom' in hw:
        a = hw['aom']
        components.append(AOMModel(
            bandwidth_MHz    = a.get('bandwidth_MHz', 10.0),
            settling_time_us = a.get('settling_time_us', 0.15),
            phase_overshoot  = a.get('phase_overshoot', 0.2),
            amplitude_noise  = a.get('amplitude_noise', 0.005),
            phase_noise_rad  = a.get('phase_noise_rad', 0.005),
            seed             = seed,
        ))

    if 'fiber' in hw:
        f = hw['fiber']
        components.append(FiberModel(
            length_mm             = f.get('length_mm', 1500.0),
            dispersion_fs2_per_mm = f.get('dispersion_fs2_per_mm', 300.0),
            length_fluctuation_nm = f.get('length_fluctuation_nm', 2.0),
            wavelength_nm         = lab_cfg.get('wavelength_nm', 302.0),
            seed                  = seed + 1,
        ))

    if not components:
        print("  Warning: no hardware components defined in lab config.")

    return HardwarePipeline(components)


# ---------------------------------------------------------------------------
# Single-qubit GRAPE
# ---------------------------------------------------------------------------

def run_single_qubit(lab_cfg, gate_cfg, args):
    # Prioritize gate config, fallback to lab config
    N  = gate_cfg.get('N_timesteps', lab_cfg['N_timesteps'])
    duration = gate_cfg.get('gate_duration_us', lab_cfg['gate_duration_us'])
    dt = duration / N

    H_kwargs = {
        'delta_r'  : lab_cfg['delta_r_MHz'],
        'delta_m'  : lab_cfg.get('delta_m_MHz', 0.0),
        'cg_strong': lab_cfg['cg_strong'],
        'cg_weak'  : lab_cfg['cg_weak'],
    }

    U_target, qubit_indices = build_target_gate(gate_cfg)

    opt_cfg = gate_cfg.get('optimizer', {})
    init_cfg = gate_cfg.get('init', {})

    print(f"  Gate: {gate_cfg['gate_name']}  |  "
          f"N={N} timesteps  |  dt={dt:.4f} us")
    print(f"  Target: {gate_cfg['target']}  |  "
          f"Method: {opt_cfg.get('method', 'lbfgs')}")

    n_starts = opt_cfg.get('n_starts', 1)

    if n_starts > 1:
        print(f"  Running {n_starts} parallel starts...")
        best, all_results = grape_multistart(
            build_H, dt, U_target, qubit_indices,
            H_kwargs    = H_kwargs,
            N_timesteps = N,
            n_starts    = n_starts,
            optimizer   = opt_cfg.get('method', 'lbfgs'),
            opt_kwargs  = {
                'max_iter': opt_cfg.get('max_iter', 500),
                'tol'     : opt_cfg.get('tol', 1e-9),
            },
            Omega_mean  = init_cfg.get('Omega_mean', 1.6),
            Omega_std   = init_cfg.get('Omega_std', 0.2),
            verbose     = args.verbose,
        )
    else:
        Omega_init, phi_init = initialize_pulse(
            N,
            Omega_mean = init_cfg.get('Omega_mean', 1.6),
            Omega_std  = init_cfg.get('Omega_std', 0.2),
            seed       = 0,
        )
        best = grape_lbfgs(
            Omega_init, phi_init, build_H, dt,
            U_target, qubit_indices, H_kwargs,
            max_iter = opt_cfg.get('max_iter', 500),
            tol      = opt_cfg.get('tol', 1e-9),
            verbose  = args.verbose,
        )

    return best, dt, N


# ---------------------------------------------------------------------------
# Two-qubit GRAPE
# ---------------------------------------------------------------------------

def run_two_qubit(lab_cfg, gate_cfg, args):
    """Run GRAPE optimization for the CZ gate."""
    from scipy.optimize import minimize

    # Prioritize gate config, fallback to lab config
    N  = gate_cfg.get('N_timesteps', lab_cfg['N_timesteps'])
    duration = gate_cfg.get('gate_duration_us', lab_cfg['gate_duration_us'])
    dt = duration / N

    delta_r   = lab_cfg['delta_r_MHz']
    delta_m   = lab_cfg.get('delta_m_MHz', 0.0)
    cg_strong = lab_cfg['cg_strong']
    cg_weak   = lab_cfg['cg_weak']

    init_cfg = gate_cfg.get('init', {})
    opt_cfg  = gate_cfg.get('optimizer', {})
    n_starts = opt_cfg.get('n_starts', 4)

    print(f"  Gate: CZ (two-qubit Rydberg blockade)")
    print(f"  N={N} timesteps  |  dt={dt:.4f} us  |  {n_starts} starts")

    def objective(x):
        Omega = x[:N]
        phi   = x[N:]
        grad_O, grad_p, F = compute_two_qubit_gradients(
            Omega, phi, delta_r, dt, delta_m, cg_strong, cg_weak)
        L      = 1.0 - F
        grad_L = np.concatenate([grad_O, grad_p])
        return L, grad_L

    best_result = None
    best_F      = -1.0

    for start in range(n_starts):
        Omega_init, phi_init = initialize_pulse(
            N,
            Omega_mean = init_cfg.get('Omega_mean', 1.6),
            Omega_std  = init_cfg.get('Omega_std', 0.2),
            seed       = start,
        )
        x0 = np.concatenate([Omega_init, phi_init])
        bounds = [(1e-6, None)] * N + [(None, None)] * N

        t0  = time.time()
        t0  = time.time()
        
        if getattr(args, 'basinhopping', False):
            from scipy.optimize import basinhopping
            minimizer_kwargs = {
                "method": "L-BFGS-B", 
                "jac": True, 
                "bounds": bounds,
                "options": {'maxiter': opt_cfg.get('max_iter', 1000), 
                            'ftol': opt_cfg.get('tol', 1e-8), 
                            'gtol': opt_cfg.get('tol', 1e-8)}
            }
            res = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs,
                               niter=opt_cfg.get('n_iter', 50),
                               T=opt_cfg.get('T', 0.15))
            
            is_success = res.lowest_optimization_result.success
            msg = res.lowest_optimization_result.message
            n_iters = res.lowest_optimization_result.nit
            
        else:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': opt_cfg.get('max_iter', 1000),
                                    'ftol'   : opt_cfg.get('tol', 1e-8),
                                    'gtol'   : opt_cfg.get('tol', 1e-8)})
            is_success = res.success
            msg = res.message
            n_iters = res.nit

        F = 1.0 - res.fun
        if args.verbose:
            print(f"  start {start}: F={F:.6f}  "
                  f"({'converged' if is_success else 'not converged'})")

        if F > best_F:
            best_F = F
            Omega_opt, phi_opt = res.x[:N], res.x[N:]
            best_result = GRAPEResult(
                Omega_opt        = Omega_opt,
                phi_opt          = phi_opt,
                fidelity         = F,
                infidelity       = 1.0 - F,
                fidelity_history = [],
                n_iterations     = n_iters,     
                converged        = is_success,  
                runtime_s        = time.time() - t0,
                message          = msg,      
            )

    return best_result, dt, N


# ---------------------------------------------------------------------------
# Hardware correction
# ---------------------------------------------------------------------------

def run_hardware_correction(result, lab_cfg, gate_cfg, dt, N, args):
    """Apply hardware pipeline correction to the optimized pulse."""
    hw_cfg = gate_cfg.get('hardware_correction', {})

    if not hw_cfg.get('enabled', True) or args.nohardware:
        print("  Hardware correction: skipped")
        return None, None

    print("  Building hardware pipeline...")
    pipeline = build_hardware_pipeline(lab_cfg)
    pipeline.summary()

    duration_us = dt * N
    ideal_pulse = Pulse.from_grape(
        result.Omega_opt, result.phi_opt, duration_us)

    # what does hardware do to the ideal pulse?
    distorted = pipeline.forward(ideal_pulse)
    loss_before = distorted.loss_vs(ideal_pulse)
    print(f"  Loss before correction: {loss_before:.6f}")

    # find input that produces ideal pulse at output
    print("  Running hardware correction (Adam optimizer)...")
    corrected_input, loss_hist = pipeline.correct(
        ideal_pulse,
        n_iter        = hw_cfg.get('n_iter', 300),
        learning_rate = hw_cfg.get('learning_rate', 0.05),
        verbose       = args.verbose,
    )

    # verify
    final_out  = pipeline.forward(corrected_input)
    loss_after = final_out.loss_vs(ideal_pulse)
    print(f"  Loss after  correction: {loss_after:.8f}  "
          f"({loss_before/max(loss_after,1e-12):.0f}x improvement)")

    return corrected_input, loss_hist


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def make_output_path(lab_cfg, gate_cfg, args):
    if args.output:
        return args.output

    os.makedirs('results', exist_ok=True)
    lab_name  = os.path.splitext(os.path.basename(args.lab))[0]
    gate_name = os.path.splitext(os.path.basename(args.gate))[0]
    ts        = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"results/{lab_name}_{gate_name}_{ts}.json"


def save_results(path, lab_cfg, gate_cfg, result,
                 corrected_pulse, hw_loss_hist, total_time):
    """Save everything to a JSON file."""
    data = {
        'timestamp'  : datetime.now().isoformat(),
        'runtime_s'  : total_time,
        'lab_config' : lab_cfg,
        'gate_config': gate_cfg,
        'grape': {
            'fidelity'        : result.fidelity,
            'infidelity'      : result.infidelity,
            'converged'       : result.converged,
            'n_iterations'    : result.n_iterations,
            'grape_runtime_s' : result.runtime_s,
            'message'         : result.message,
            'Omega_opt'       : result.Omega_opt.tolist(),
            'phi_opt'         : result.phi_opt.tolist(),
            'fidelity_history': result.fidelity_history,
        },
        'hardware_correction': {
            'applied'         : corrected_pulse is not None,
            'Omega_corrected' : corrected_pulse.Omega.tolist()
                                if corrected_pulse is not None else None,
            'phi_corrected'   : corrected_pulse.phi.tolist()
                                if corrected_pulse is not None else None,
            'loss_history'    : hw_loss_hist
                                if hw_loss_hist is not None else None,
        }
    }

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n  Results saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GRAPE pulse optimizer for neutral atom qubits')
    parser.add_argument('-lab', '--lab', required=True,
                        help='Lab config YAML (e.g. labs/real.yaml)')
    parser.add_argument('-gate', '--gate', required=True,
                        help='Gate config YAML (e.g. gates/cz.yaml)')
    parser.add_argument('-output', '--output', default=None,
                        help='Output JSON path (default: auto-generated in results/)')
    parser.add_argument('-nohardware', '--no-hardware', dest='nohardware', action='store_true',
                        help='Skip hardware correction step')
    parser.add_argument('-verbose', '--verbose', action='store_true',
                        help='Print optimization progress')
    parser.add_argument('-basinhopping', '--basinhopping', action="store_true", help="Use Basin Hopping global optimizer")
    
    args = parser.parse_args()

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  GRAPE Pulse Optimizer")
    print(f"{'='*60}")
    print(f"  Lab:  {args.lab}")
    print(f"  Gate: {args.gate}")
    print()

    # load configs
    lab_cfg  = load_yaml(args.lab)
    gate_cfg = load_yaml(args.gate)
    print(f"  Lab:  {lab_cfg['lab_name']}")
    print(f"  Gate: {gate_cfg['gate_name']}  --  {gate_cfg['description']}")
    print()

    # run GRAPE
    print("--- GRAPE Optimization ---")
    gate_type = gate_cfg.get('gate_type', 'single_qubit')
    if gate_type == 'two_qubit':
        result, dt, N = run_two_qubit(lab_cfg, gate_cfg, args)
    else:
        result, dt, N = run_single_qubit(lab_cfg, gate_cfg, args)

    print()
    print("  GRAPE result:")
    result.summary()

    # hardware correction
    print()
    print("--- Hardware Correction ---")
    corrected_pulse, hw_loss_hist = run_hardware_correction(
        result, lab_cfg, gate_cfg, dt, N, args)

    # save
    total_time  = time.time() - t_start
    output_path = make_output_path(lab_cfg, gate_cfg, args)
    save_results(output_path, lab_cfg, gate_cfg, result,
                 corrected_pulse, hw_loss_hist, total_time)

    print()
    print(f"{'='*60}")
    print(f"  Done in {total_time:.1f}s")
    print(f"  Final fidelity: {result.fidelity:.8f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

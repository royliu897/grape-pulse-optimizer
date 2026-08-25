"""
Comprehensive test suite.

Run with: python tests/test_all.py

Tests are organized by module and check both correctness
(VJPs vs numerical) and functionality (optimization converges).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hardware.pipeline import Pulse, HardwareComponent, HardwarePipeline
from hardware.components import AOMModel, FiberModel
from physics.systems.rydberg import build_H
from physics.systems.two_qubit_rydberg import build_H_two_qubit
from physics.propagator import compute_propagators, pulse_to_H_list
from grape.fidelity import gate_fidelity, infidelity, target_CZ
from grape.gradients import compute_gradients, numerical_gradient
from grape.optimizer import grape_lbfgs, initialize_pulse
from physics.two_qubit_propagator import two_qubit_gate_fidelity


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Hardware tests
# ---------------------------------------------------------------------------

def test_aom_vjp():
    N = 30
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t,
        1.6*np.exp(-((t-0.6)**2)/(2*0.25**2)),
        np.linspace(0, np.pi, N))
    np.random.seed(0)
    g_out = np.random.randn(N) + 1j*np.random.randn(N)

    aom = AOMModel(bandwidth_MHz=12, settling_time_us=0.15,
                   phase_overshoot=0.2, amplitude_noise=0.0,
                   phase_noise_rad=0.0, seed=0)
    g_an  = aom.vjp(target, g_out)
    g_num = HardwareComponent.vjp(aom, target, g_out)
    err = np.max(np.abs(g_an - g_num))
    assert err < 1e-4, f"AOM VJP error {err:.2e} > 1e-4"


def test_fiber_vjp():
    N = 30
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t, np.ones(N)*1.6, np.linspace(0, np.pi, N))
    np.random.seed(1)
    g_out = np.random.randn(N) + 1j*np.random.randn(N)

    fiber = FiberModel(length_mm=1500, dispersion_fs2_per_mm=300,
                       length_fluctuation_nm=0.0, seed=0)
    g_an  = fiber.vjp(target, g_out)
    g_num = HardwareComponent.vjp(fiber, target, g_out)
    err = np.max(np.abs(g_an - g_num))
    assert err < 1e-4, f"Fiber VJP error {err:.2e} > 1e-4"


def test_gradient_points_downhill():
    N = 60
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t,
        1.6*np.exp(-((t-0.6)**2)/(2*0.25**2)),
        np.linspace(0, np.pi, N))

    pipeline = HardwarePipeline([
        AOMModel(bandwidth_MHz=12, settling_time_us=0.15,
                 phase_overshoot=0.2, amplitude_noise=0.0,
                 phase_noise_rad=0.0, seed=0)
    ])
    out   = pipeline.forward(target)
    loss0 = out.loss_vs(target)
    g_l   = out.grad_of_loss_vs(target)
    g_in  = pipeline.backpropagate(target, g_l)

    A_new = target.A - 0.05 * g_in
    loss1 = pipeline.forward(Pulse(target.t, A_new)).loss_vs(target)
    assert loss1 < loss0, f"Gradient step increased loss: {loss0:.6f} -> {loss1:.6f}"


def test_aom_correction_converges():
    N = 60
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t,
        1.6*np.exp(-((t-0.6)**2)/(2*0.25**2)),
        np.linspace(0, np.pi, N))
    pipeline = HardwarePipeline([
        AOMModel(bandwidth_MHz=12, settling_time_us=0.15,
                 phase_overshoot=0.2, amplitude_noise=0.0,
                 phase_noise_rad=0.0, seed=0)
    ])
    raw_loss = pipeline.forward(target).loss_vs(target)
    corrected, hist = pipeline.correct(target, n_iter=200)
    final_loss = pipeline.forward(corrected).loss_vs(target)
    assert final_loss < raw_loss * 0.1, \
        f"AOM correction: {raw_loss:.4f} -> {final_loss:.4f} (not 10x improvement)"


def test_full_pipeline_correction():
    N = 60
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t,
        1.6*np.exp(-((t-0.6)**2)/(2*0.25**2)),
        np.linspace(0, np.pi, N))
    pipeline = HardwarePipeline([
        AOMModel(bandwidth_MHz=15, settling_time_us=0.15,
                 phase_overshoot=0.2, amplitude_noise=0.0,
                 phase_noise_rad=0.0, seed=1),
        FiberModel(length_mm=1500, dispersion_fs2_per_mm=500,
                   length_fluctuation_nm=0.0, seed=1)
    ])
    raw_loss = pipeline.forward(target).loss_vs(target)
    corrected, _ = pipeline.correct(target, n_iter=300)
    final_loss = pipeline.forward(corrected).loss_vs(target)
    assert final_loss < raw_loss * 0.1, \
        f"Pipeline correction: {raw_loss:.4f} -> {final_loss:.4f}"


def test_component_swap():
    N = 60
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t, np.ones(N)*1.6, np.linspace(0, np.pi, N))
    pipeline = HardwarePipeline([
        AOMModel(bandwidth_MHz=15, settling_time_us=0.1,
                 phase_overshoot=0.1, amplitude_noise=0.0,
                 phase_noise_rad=0.0, seed=0),
        FiberModel(length_mm=1500, dispersion_fs2_per_mm=300,
                   length_fluctuation_nm=0.0, seed=0)
    ])
    _, hist1 = pipeline.correct(target, n_iter=100)

    # swap fiber
    pipeline.remove_component(1)
    pipeline.add_component(FiberModel(length_mm=500, dispersion_fs2_per_mm=100,
                                       length_fluctuation_nm=0.0, seed=1))
    _, hist2 = pipeline.correct(target, n_iter=100)
    assert hist2[-1] < hist2[0], "Correction should decrease loss after swap"


# ---------------------------------------------------------------------------
# GRAPE single-qubit tests
# ---------------------------------------------------------------------------

def test_single_qubit_grape_gradients():
    delta_r = 9.3
    N  = 15
    dt = 1.2 / N
    qubit_indices = [0, 1]
    H_kwargs = {'delta_r': delta_r}
    U_target = np.eye(2, dtype=complex)

    np.random.seed(42)
    Omega = np.ones(N)*1.6 + np.random.randn(N)*0.05
    phi   = np.random.uniform(-0.1, 0.1, N)

    g_an, gp_an, F = compute_gradients(
        Omega, phi, build_H, dt, U_target, qubit_indices, H_kwargs)
    g_num, gp_num = numerical_gradient(
        Omega.copy(), phi.copy(), build_H, dt, U_target, qubit_indices, H_kwargs)

    err_O = np.max(np.abs(g_an - g_num))
    err_p = np.max(np.abs(gp_an - gp_num))
    assert err_O < 1e-6, f"Single-qubit Omega gradient error: {err_O:.2e}"
    assert err_p < 1e-6, f"Single-qubit phi gradient error: {err_p:.2e}"


def test_single_qubit_optimization():
    delta_r = 9.3
    N  = 20
    dt = 1.2 / N
    H_kwargs = {'delta_r': delta_r}
    U_target = np.eye(2, dtype=complex)

    Omega_init, phi_init = initialize_pulse(N, seed=0)
    result = grape_lbfgs(
        Omega_init, phi_init, build_H, dt,
        U_target, [0, 1], H_kwargs, max_iter=200, verbose=False)
    assert result.fidelity > 0.99, \
        f"Single-qubit optimization fidelity {result.fidelity:.4f} < 0.99"


# ---------------------------------------------------------------------------
# Two-qubit tests
# ---------------------------------------------------------------------------

def test_two_qubit_hamiltonian_hermitian():
    H_dict = build_H_two_qubit(1.6, 0.5, 9.3)
    for sub, H in H_dict.items():
        diff = np.max(np.abs(H - H.conj().T))
        assert diff < 1e-14, f"H_{sub} not Hermitian: max diff {diff:.2e}"


def test_two_qubit_propagator_unitary():
    from physics.two_qubit_propagator import (
        build_H_list_two_qubit, compute_two_qubit_propagators)
    N  = 10
    dt = 1.2 / N
    Omega = np.ones(N) * 1.6
    phi   = np.zeros(N)
    H_lists = build_H_list_two_qubit(Omega, phi, 9.3)
    U_dict  = compute_two_qubit_propagators(H_lists, dt)
    for sub, U in U_dict.items():
        d    = U.shape[0]
        prod = U.conj().T @ U
        diff = np.max(np.abs(prod - np.eye(d)))
        assert diff < 1e-12, f"U_{sub} not unitary: {diff:.2e}"


def test_two_qubit_fidelity_range():
    N  = 10
    dt = 1.2 / N
    Omega = np.ones(N) * 1.6
    phi   = np.zeros(N)
    F, elements = two_qubit_gate_fidelity(Omega, phi, 9.3, dt)
    assert 0 <= F <= 1, f"Fidelity {F} out of range [0,1]"


if __name__ == '__main__':
    print("=== Hardware tests ===")
    run_test("AOM VJP vs numerical",          test_aom_vjp)
    run_test("Fiber VJP vs numerical",         test_fiber_vjp)
    run_test("Gradient points downhill",       test_gradient_points_downhill)
    run_test("AOM correction converges",       test_aom_correction_converges)
    run_test("Full pipeline correction",       test_full_pipeline_correction)
    run_test("Component swap works",           test_component_swap)

    print()
    print("=== Single-qubit GRAPE tests ===")
    run_test("Gradient vs numerical",          test_single_qubit_grape_gradients)
    run_test("Optimization converges",         test_single_qubit_optimization)

    print()
    print("=== Two-qubit tests ===")
    run_test("Hamiltonian Hermitian",          test_two_qubit_hamiltonian_hermitian)
    run_test("Propagator unitary",             test_two_qubit_propagator_unitary)
    run_test("Fidelity in [0,1]",             test_two_qubit_fidelity_range)

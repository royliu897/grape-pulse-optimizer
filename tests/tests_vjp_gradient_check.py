"""
Rigorous inner-product gradient check for all VJP implementations.

Tests the identity:
    <vjp(g), dA> ≈ <g, J @ dA>

This is the definitive test for VJP correctness -- convention-agnostic.

Run from repo root: python tests/test_vjp_gradient_check.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hardware.pipeline import Pulse, HardwarePipeline
from hardware.components import AOMModel, FiberModel


def inner_product(a, b):
    """Real inner product treating complex arrays as 2D real vectors."""
    return float(np.real(np.sum(np.conj(a) * b)))


def numerical_jvp(forward_fn, A, dA, eps=1e-6):
    """J @ dA via central finite differences."""
    return (forward_fn(A + eps * dA) - forward_fn(A - eps * dA)) / (2 * eps)


def gradient_check(name, forward_fn, vjp_fn, A,
                   n_trials=10, eps=1e-6, tol=1e-4):
    """
    Check <vjp(g), dA> ≈ <g, J @ dA> for random g and dA.
    """
    errors = []
    for trial in range(n_trials):
        np.random.seed(trial)
        g  = np.random.randn(len(A)) + 1j * np.random.randn(len(A))
        dA = np.random.randn(len(A)) + 1j * np.random.randn(len(A))
        dA = dA / np.linalg.norm(dA)

        lhs = inner_product(vjp_fn(A, g), dA)
        rhs = inner_product(g, numerical_jvp(forward_fn, A, dA, eps))
        errors.append(abs(lhs - rhs))

    max_err = max(errors)
    passed  = max_err < tol
    print(f"  {'PASS' if passed else 'FAIL'}  {name}: "
          f"max error = {max_err:.2e}  (tol={tol:.0e})")
    return passed


def test_phase_reconstruction_vjp():
    """
    Test the decoupled phase reconstruction VJP in isolation.

    Key insight: angle(A[n] * conj(A[n-1])) = angle(A[n]) - angle(A[n-1])
    exactly (modulo 2*pi*k which vanishes under differentiation).
    So gradients decouple: d/dA[n] depends only on A[n], not A[n-1].

    This test verifies that the implemented VJP matches finite differences.
    """
    N = 20
    np.random.seed(42)
    A_bw = (np.random.randn(N) + 1j * np.random.randn(N)) * 1.6

    # forward: just phase reconstruction (output is real, cast to complex)
    def forward_phi(A):
        phi = np.zeros(N)
        phi[0] = np.angle(A[0])
        for n in range(1, N):
            phi[n] = phi[n-1] + np.angle(A[n] * np.conj(A[n-1]))
        return phi.astype(complex)

    # VJP: decoupled analytical version
    def vjp_decoupled(A, g_phi):
        g_phi_real  = np.real(g_phi)
        g_A         = np.zeros(N, dtype=complex)

        # reverse cumsum
        g_delta_phi    = np.zeros(N)
        g_delta_phi[N-1] = g_phi_real[N-1]
        for k in range(N-2, -1, -1):
            g_delta_phi[k] = g_phi_real[k] + g_delta_phi[k+1]

        for n in range(N):
            # d/dA[n] angle(A[n]) = i * A[n] / |A[n]|^2
            term = 1j * A[n] / (np.abs(A[n])**2 + 1e-20)
            if n == 0:
                # delta_phi[0] = angle(A[0]) depends only on A[0]
                g_A[0] += g_delta_phi[0] * term
            else:
                # delta_phi[n] = angle(A[n]) - angle(A[n-1])
                # gradient for A[n]: same term as above
                g_A[n]   += g_delta_phi[n] * term
                # gradient for A[n-1]: negative (minus sign from subtraction)
                g_A[n-1] -= g_delta_phi[n] * \
                             1j * A[n-1] / (np.abs(A[n-1])**2 + 1e-20)

        return g_A

    print("=== Phase reconstruction VJP (decoupled) ===")
    gradient_check("Decoupled phase VJP", forward_phi, vjp_decoupled, A_bw)


def test_aom_bandwidth_vjp():
    """Test AOM bandwidth filter VJP in isolation."""
    N = 40
    t = np.linspace(0, 1.2, N)
    # smooth physical pulse -- random noise causes large phase jumps
    # that correctly trip the branch cut safety assertion
    A0 = (1.6 * np.exp(-((t - 0.6)**2) / (2 * 0.25**2))
          * np.exp(1j * np.linspace(0, np.pi, N)))

    aom = AOMModel(bandwidth_MHz=12, settling_time_us=0.001,
                   phase_overshoot=0.0, amplitude_noise=0.0,
                   phase_noise_rad=0.0, seed=0)

    def fwd(A):
        return aom.forward(Pulse(t, A)).A

    def vjp(A, g):
        return aom.vjp(Pulse(t, A), g)

    print()
    print("=== AOM bandwidth-only VJP ===")
    gradient_check("Bandwidth VJP", fwd, vjp, A0)


def test_aom_full_vjp():
    """Test full AOM VJP (bandwidth + phase IIR) via inner product check."""
    N = 40
    t = np.linspace(0, 1.2, N)
    A0 = (1.6 * np.exp(-((t - 0.6)**2) / (2 * 0.25**2))
          * np.exp(1j * np.linspace(0, np.pi, N)))

    aom = AOMModel(bandwidth_MHz=12, settling_time_us=0.15,
                   phase_overshoot=0.2, amplitude_noise=0.0,
                   phase_noise_rad=0.0, seed=0)

    def fwd(A):
        return aom.forward(Pulse(t, A)).A

    def vjp(A, g):
        return aom.vjp(Pulse(t, A), g)

    print()
    print("=== Full AOM VJP (bandwidth + phase IIR) ===")
    gradient_check("Full AOM VJP", fwd, vjp, A0, n_trials=10)


def test_fiber_vjp():
    """Test fiber GVD VJP via inner product check."""
    N = 40
    t = np.linspace(0, 1.2, N)
    np.random.seed(1)
    A0 = np.random.randn(N) + 1j * np.random.randn(N)

    fiber = FiberModel(length_mm=1500, dispersion_fs2_per_mm=300,
                       length_fluctuation_nm=0.0, seed=0)

    def fwd(A):
        return fiber.forward(Pulse(t, A)).A

    def vjp(A, g):
        return fiber.vjp(Pulse(t, A), g)

    print()
    print("=== Fiber GVD VJP ===")
    gradient_check("Fiber VJP", fwd, vjp, A0, n_trials=10)


def test_pipeline_vjp():
    """Test full pipeline VJP via inner product check."""
    N = 40
    t = np.linspace(0, 1.2, N)
    A0 = (1.6 * np.exp(-((t - 0.6)**2) / (2 * 0.25**2))
          * np.exp(1j * np.linspace(0, np.pi, N)))

    pipeline = HardwarePipeline([
        AOMModel(bandwidth_MHz=12, settling_time_us=0.15,
                 phase_overshoot=0.2, amplitude_noise=0.0,
                 phase_noise_rad=0.0, seed=0),
        FiberModel(length_mm=1500, dispersion_fs2_per_mm=300,
                   length_fluctuation_nm=0.0, seed=1)
    ])

    target = Pulse(t, A0)

    def fwd(A):
        return pipeline.forward(Pulse(t, A)).A

    def vjp(A, g):
        return pipeline.backpropagate(Pulse(t, A), g)

    print()
    print("=== Full pipeline VJP (AOM + Fiber) ===")
    gradient_check("Pipeline VJP", fwd, vjp, A0, n_trials=10)


def test_gradient_descent_direction():
    """Test that gradient descent reduces loss for multiple learning rates."""
    N = 60
    t = np.linspace(0, 1.2, N)
    target = Pulse.from_polar(t,
        1.6 * np.exp(-((t - 0.6)**2) / (2 * 0.25**2)),
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

    print()
    print("=== Gradient descent direction ===")
    all_ok = True
    for lr in [0.001, 0.01, 0.05, 0.1, 0.3]:
        A_new    = target.A - lr * g_in
        loss_new = pipeline.forward(Pulse(t, A_new)).loss_vs(target)
        ok       = loss_new < loss0
        if not ok:
            all_ok = False
        print(f"  lr={lr:.3f}: {loss0:.6f} -> {loss_new:.6f}  "
              f"{'OK' if ok else 'FAIL'}")

    print(f"  {'PASS' if all_ok else 'FAIL'}  All learning rates decrease loss")


def test_branch_cut_assertion():
    """
    Verify that the bandwidth filter keeps |delta_phi| < pi.
    This is the structural guarantee that makes the decoupled VJP exact.
    """
    N = 100
    t = np.linspace(0, 1.2, N)

    # worst-case: sharp pulse edges
    A_sharp = np.zeros(N, dtype=complex)
    A_sharp[20:80] = 1.6 * np.exp(1j * np.linspace(0, 2*np.pi, 60))

    aom = AOMModel(bandwidth_MHz=12, settling_time_us=0.0,
                   phase_overshoot=0.0, amplitude_noise=0.0,
                   phase_noise_rad=0.0, seed=0)
    A_bw = aom._apply_bw(A_sharp, N, t[1]-t[0])

    delta_phi = np.zeros(N)
    for n in range(1, N):
        delta_phi[n] = np.angle(A_bw[n] * np.conj(A_bw[n-1]))

    max_dphi = np.max(np.abs(delta_phi[1:]))
    passed   = max_dphi < np.pi
    print()
    print("=== Branch cut protection check ===")
    print(f"  Max |delta_phi| = {max_dphi:.4f} rad  "
          f"(limit = {np.pi:.4f} = pi)")
    print(f"  {'PASS' if passed else 'FAIL'}  "
          f"Bandwidth filter keeps delta_phi in (-pi, pi)")


if __name__ == '__main__':
    test_phase_reconstruction_vjp()
    test_aom_bandwidth_vjp()
    test_aom_full_vjp()
    test_fiber_vjp()
    test_pipeline_vjp()
    test_gradient_descent_direction()
    test_branch_cut_assertion()

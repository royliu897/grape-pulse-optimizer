"""
Propagator module.

Takes a Hamiltonian H (or sequence of Hamiltonians) and computes
the unitary evolution operator U.

For a single timestep with constant H:
    U_k = expm(-i * H_k * dt)

For a full pulse sequence of N timesteps:
    U_total = U_N @ U_{N-1} @ ... @ U_2 @ U_1

Also computes and stores the forward and backward propagators
needed by GRAPE to compute gradients efficiently:

    Forward:  F_k = U_k @ U_{k-1} @ ... @ U_1   (state at end of step k)
    Backward: B_k = U_N @ U_{N-1} @ ... @ U_{k+1} (what remains after step k)

Units: everything in 2*pi*MHz and microseconds.
       With hbar=1, the exponent is just -i * H * dt directly.
"""

import numpy as np
from scipy.linalg import expm


def compute_U_single(H, dt):
    """
    Compute the unitary propagator for a single timestep.

    U = expm(-i * H * dt)

    Parameters
    ----------
    H : numpy array, shape (d, d), complex
        Hamiltonian at this timestep. Must be Hermitian.
    dt : float
        Duration of this timestep in microseconds.
        With hbar=1 and frequencies in 2pi*MHz, units work out directly.

    Returns
    -------
    U : numpy array, shape (d, d), complex
        Unitary propagator for this timestep.

    Notes
    -----
    We multiply H by 2*pi because our frequencies are in MHz
    (not angular frequency). So the exponent is:
        -i * (2*pi * H_in_MHz) * dt_in_us
    which gives a dimensionless exponent as required.

    Actually -- looking at eq (6), the paper already writes H with
    the hbar factored out and uses angular frequencies (2pi*MHz).
    So the exponent is just -i * H * dt. We keep it simple.
    """
    d = H.shape[0]
    exponent = -1j * H * dt
    U = expm(exponent)
    return U


def check_unitary(U, tol=1e-10):
    """
    Check that a matrix is unitary: U† @ U = I, thus that
    probabilites(magnitude of state vector) must add up to 1.

    Parameters
    ----------
    U : numpy array, square complex
    tol : float
        Tolerance for deviation from identity.

    Returns
    -------
    is_unitary : bool
    max_deviation : float
        Maximum element-wise deviation of U†U from identity.
    """
    d = U.shape[0]
    product = U.conj().T @ U
    identity = np.eye(d, dtype=complex)
    max_dev = np.max(np.abs(product - identity))
    return max_dev < tol, max_dev


def compute_forward_propagators(H_list, dt):
    """
    Compute all forward propagators for a pulse sequence.

    F_k = U_k @ U_{k-1} @ ... @ U_1

    F_k represents the total evolution from the start up to
    and including timestep k.

    F_0 is defined as the identity (nothing has happened yet).
    F_N is U_total (the full gate).

    Parameters
    ----------
    H_list : list of numpy arrays, each shape (d, d)
        Hamiltonians at each timestep. Length N.
    dt : float
        Duration of each timestep in microseconds.

    Returns
    -------
    F : list of numpy arrays, length N+1
        F[0] = identity
        F[k] = U_k @ F[k-1]   for k = 1..N
        F[N] = U_total

    Notes
    -----
    We store F[0] through F[N] so that F[k-1] is always
    available when computing the gradient at step k.
    This costs memory (N+1 matrices) but is essential for
    efficient gradient computation -- we never recompute these.
    """
    N = len(H_list)
    d = H_list[0].shape[0]

    F = [None] * (N + 1)
    F[0] = np.eye(d, dtype=complex)   # identity -- nothing happened yet

    for k in range(N):
        U_k = compute_U_single(H_list[k], dt)
        F[k + 1] = U_k @ F[k]        # chain: apply U_k after everything so far

    return F


def compute_backward_propagators(H_list, dt):
    """
    Compute all backward propagators for a pulse sequence.

    B_k = U_N @ U_{N-1} @ ... @ U_{k+1}

    B_k represents the total evolution from just after timestep k
    to the end of the pulse.

    B_N is defined as the identity (nothing remains after the last step).
    B_0 is U_total (everything remains to be applied).

    Parameters
    ----------
    H_list : list of numpy arrays, each shape (d, d)
        Hamiltonians at each timestep. Length N.
    dt : float
        Duration of each timestep in microseconds.

    Returns
    -------
    B : list of numpy arrays, length N+1
        B[N] = identity
        B[k] = B[k+1] @ U_{k+1}   for k = N-1..0
        B[0] = U_total

    Notes
    -----
    We sweep backward through the timesteps, accumulating U matrices
    from right to left. This is the "backward pass" that GRAPE needs.
    """
    N = len(H_list)
    d = H_list[0].shape[0]

    # first compute all individual U_k matrices
    # we need them for both forward and backward passes
    U_list = [compute_U_single(H_list[k], dt) for k in range(N)]

    B = [None] * (N + 1)
    B[N] = np.eye(d, dtype=complex)   # identity -- nothing remains after last step

    for k in range(N - 1, -1, -1):   # sweep backward: N-1, N-2, ..., 0
        B[k] = B[k + 1] @ U_list[k]  # chain: apply U_k before everything remaining

    return B


def compute_propagators(H_list, dt):
    """
    Compute both forward and backward propagators in one call.

    This is the main function called by the GRAPE gradient computation.
    It returns everything needed to compute gradients at every timestep
    without any redundant matrix multiplications.

    Parameters
    ----------
    H_list : list of numpy arrays, each shape (d, d)
        Hamiltonians at each timestep. Length N.
    dt : float
        Duration of each timestep in microseconds.

    Returns
    -------
    F : list of N+1 forward propagators
        F[k] = evolution from start through step k
    B : list of N+1 backward propagators
        B[k] = evolution from step k+1 through end
    U_total : numpy array, shape (d, d)
        The full gate unitary = F[N] = B[0]

    Notes
    -----
    U_total can be read from either F[N] or B[0] -- they should
    be identical. We return F[N] as U_total.
    """
    F = compute_forward_propagators(H_list, dt)
    B = compute_backward_propagators(H_list, dt)
    U_total = F[-1]   # F[N] = full gate
    return F, B, U_total


def pulse_to_H_list(Omega_list, phi_list, build_H_func, **H_kwargs):
    """
    Convert pulse parameter arrays into a list of Hamiltonians.

    This bridges the GRAPE optimizer (which works with arrays of
    Omega and phi values) and the propagator (which works with
    a list of H matrices).

    Parameters
    ----------
    Omega_list : array-like, length N
        Rabi frequency at each timestep.
    phi_list : array-like, length N
        Laser phase at each timestep.
    build_H_func : callable
        Function with signature build_H(Omega, phi, **kwargs) -> H matrix.
        Typically physics.systems.rydberg.build_H
    **H_kwargs :
        Additional fixed parameters passed to build_H_func
        (e.g. delta_r, cg_strong, etc.)

    Returns
    -------
    H_list : list of N numpy arrays, each shape (d, d)
    """
    return [
        build_H_func(Omega, phi, **H_kwargs)
        for Omega, phi in zip(Omega_list, phi_list)
    ]

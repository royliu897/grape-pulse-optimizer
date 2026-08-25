"""
Gradient module.

Computes the gradient of the infidelity L = 1 - F with respect
to every pulse parameter (Omega_k, phi_k) at every timestep k.

The key formula is:

    dL/du_k = -(2/d^2) * Re[ overlap* * Tr(U_target† @ B_k @ dU_k/du_k @ F_{k-1}) ]

where:
    overlap  = Tr(U_target† @ U_sub)   -- complex overlap from fidelity
    B_k      = backward propagator (evolution from step k+1 to end)
    F_{k-1}  = forward propagator (evolution from start to step k-1)
    dU_k/du_k = derivative of U_k with respect to control parameter u_k

And dU_k/du_k is computed using the approximation:
    dU_k/du_k ≈ (-i * dt * dH_k/du_k) @ U_k

which is exact when dH_k/du_k commutes with H_k (approximately true
for small dt).

This module also provides numerical gradient checking via finite
differences, which is essential for verifying the analytical gradients.
"""

import numpy as np
from scipy.linalg import expm

from physics.propagator import compute_U_single, compute_propagators, pulse_to_H_list
from grape.fidelity import gate_fidelity, infidelity, fidelity_and_overlap, project_to_subspace


def dU_exact(H, dH_du, dt):
    """
    Compute the exact derivative of U = expm(-i*H*dt) with respect
    to a control parameter u, given dH/du.

    Uses the augmented matrix method:
        expm([[M, dM], [0, M]])
    where M = -i*H*dt and dM = -i*dH_du*dt.
    The off-diagonal block of the result gives the exact derivative dU/du.

    This is more accurate than the first-order approximation
        dU/du ≈ (-i*dt*dH_du) @ U
    which has O(dt^2) error and becomes inaccurate for large timesteps.

    Parameters
    ----------
    H : numpy array, shape (d, d)
        Hamiltonian at this timestep.
    dH_du : numpy array, shape (d, d)
        Derivative of H with respect to control parameter u.
    dt : float
        Timestep duration.

    Returns
    -------
    dU : numpy array, shape (d, d)
        Exact derivative dU/du.
    """
    d = H.shape[0]
    M  = -1j * dt * H
    dM = -1j * dt * dH_du

    # build 2d x 2d augmented matrix
    aug = np.zeros((2*d, 2*d), dtype=complex)
    aug[:d, :d] = M    # top-left block
    aug[:d, d:] = dM   # top-right block (the perturbation)
    aug[d:, d:] = M    # bottom-right block (same as top-left)
    # bottom-left block stays zero

    result = expm(aug)
    return result[:d, d:]   # top-right block of result = exact dU/du


def dH_dOmega(phi, cg_strong=0.5, cg_weak=None):
    """
    Derivative of the control Hamiltonian with respect to Omega.

    Since H_control = Omega * C(phi), where C(phi) is the coupling
    matrix at phase phi, we have:

        dH/dOmega = C(phi)

    This is just the control Hamiltonian with Omega = 1.

    Parameters
    ----------
    phi : float
        Current laser phase.
    cg_strong, cg_weak : float
        Clebsch-Gordan coefficients.

    Returns
    -------
    dH : 6x6 complex numpy array
    """
    if cg_weak is None:
        cg_weak = 1.0 / (2.0 * np.sqrt(3))

    dH = np.zeros((6, 6), dtype=complex)

    up   = np.exp(-1j * phi)
    down = np.exp(+1j * phi)

    # same structure as build_H_control but with Omega = 1
    dH[0, 2] = cg_strong * up;   dH[2, 0] = cg_strong * down
    dH[0, 4] = cg_weak   * up;   dH[4, 0] = cg_weak   * down
    dH[1, 3] = cg_weak   * up;   dH[3, 1] = cg_weak   * down
    dH[1, 5] = cg_strong * up;   dH[5, 1] = cg_strong * down

    return dH


def dH_dphi(Omega, phi, cg_strong=0.5, cg_weak=None):
    """
    Derivative of the control Hamiltonian with respect to phi.

    Since H_control entries have the form Omega * cg * e^(-i*phi),
    the derivative with respect to phi brings down a factor of -i
    for upper triangle and +i for lower triangle:

        d/dphi [Omega * cg * e^(-i*phi)] = -i * Omega * cg * e^(-i*phi)
        d/dphi [Omega * cg * e^(+i*phi)] = +i * Omega * cg * e^(+i*phi)

    Parameters
    ----------
    Omega : float
        Current laser amplitude.
    phi : float
        Current laser phase.
    cg_strong, cg_weak : float
        Clebsch-Gordan coefficients.

    Returns
    -------
    dH : 6x6 complex numpy array
    """
    if cg_weak is None:
        cg_weak = 1.0 / (2.0 * np.sqrt(3))

    dH = np.zeros((6, 6), dtype=complex)

    # d/dphi of e^(-i*phi) = -i * e^(-i*phi)
    # d/dphi of e^(+i*phi) = +i * e^(+i*phi)
    up   = -1j * Omega * np.exp(-1j * phi)
    down = +1j * Omega * np.exp(+1j * phi)

    dH[0, 2] = cg_strong * up;   dH[2, 0] = cg_strong * down
    dH[0, 4] = cg_weak   * up;   dH[4, 0] = cg_weak   * down
    dH[1, 3] = cg_weak   * up;   dH[3, 1] = cg_weak   * down
    dH[1, 5] = cg_strong * up;   dH[5, 1] = cg_strong * down

    return dH


def compute_gradient_single_step(k, F, B, dU_k, U_target,
                                  overlap, qubit_indices):
    """
    Compute the gradient of infidelity at a single timestep k
    for a single control parameter u (either Omega or phi).

    Formula:
        dL/du_k = -(2/d^2) * Re[ overlap* * Tr(U_target† @ B_{k+1} @ dU_k/du @ F_{k}) ]

    Parameters
    ----------
    k : int
        Timestep index (0-based).
    F : list of arrays
        Forward propagators. F[k] = evolution from start through step k.
    B : list of arrays
        Backward propagators. B[k+1] = evolution from step k+1 to end.
    dU_k : numpy array
        Exact derivative dU_k/du, computed via dU_exact().
    U_target : numpy array
        Target gate matrix.
    overlap : complex
        Tr(U_target† @ U_sub) -- precomputed from fidelity.
    qubit_indices : list of int
        Subspace indices for projection.

    Returns
    -------
    grad : float
        Gradient dL/du_k at this timestep.
    """
    d = U_target.shape[0]

    # sandwich: B_{k+1} @ dU_k @ F_{k}
    sandwich = B[k + 1] @ dU_k @ F[k]

    # project to qubit subspace
    sandwich_sub = project_to_subspace(sandwich, qubit_indices)

    # inner product with U_target†
    inner = np.trace(U_target.conj().T @ sandwich_sub)

    # gradient: dL/du = -(2/d^2) * Re[ overlap* * inner ]
    grad = -(2.0 / d**2) * np.real(np.conj(overlap) * inner)

    return grad


def compute_gradients(Omega_list, phi_list, build_H_func, dt,
                       U_target, qubit_indices, H_kwargs=None):
    """
    Compute gradients of infidelity with respect to all pulse parameters.

    This is the main function called by the GRAPE optimizer.
    It performs one forward pass, one backward pass, then computes
    all gradients in a single sweep.

    Parameters
    ----------
    Omega_list : array-like, length N
        Rabi frequency at each timestep.
    phi_list : array-like, length N
        Laser phase at each timestep.
    build_H_func : callable
        Function build_H(Omega, phi, **kwargs) -> H matrix.
    dt : float
        Timestep duration in microseconds.
    U_target : numpy array
        Target gate matrix.
    qubit_indices : list of int
        Subspace projection indices.
    H_kwargs : dict, optional
        Fixed parameters passed to build_H_func (e.g. delta_r).

    Returns
    -------
    grad_Omega : numpy array, length N
        Gradient dL/dOmega_k for each timestep.
    grad_phi : numpy array, length N
        Gradient dL/dphi_k for each timestep.
    F_current : float
        Current fidelity (computed as a byproduct, no extra cost).
    """
    if H_kwargs is None:
        H_kwargs = {}

    N = len(Omega_list)

    # build H at each timestep from current pulse parameters
    H_list = pulse_to_H_list(Omega_list, phi_list, build_H_func, **H_kwargs)

    # forward and backward passes -- the expensive part, O(N) matrix multiplications
    F, B, U_total = compute_propagators(H_list, dt)

    # compute fidelity and overlap -- needed for gradient formula
    F_val, overlap, U_sub = fidelity_and_overlap(U_total, U_target, qubit_indices)

    # initialize gradient arrays
    grad_Omega = np.zeros(N)
    grad_phi   = np.zeros(N)

    # sweep through timesteps and compute gradient at each one
    for k in range(N):
        Omega_k = Omega_list[k]
        phi_k   = phi_list[k]

        # get the CG coefficients from H_kwargs if provided
        cg_strong = H_kwargs.get('cg_strong', 0.5)
        cg_weak   = H_kwargs.get('cg_weak', 1.0 / (2.0 * np.sqrt(3)))

        # analytical derivatives of H with respect to each control
        dH_dO = dH_dOmega(phi_k, cg_strong, cg_weak)
        dH_dp = dH_dphi(Omega_k, phi_k, cg_strong, cg_weak)

        # exact derivatives of U_k using augmented matrix method
        dU_dO = dU_exact(H_list[k], dH_dO, dt)
        dU_dp = dU_exact(H_list[k], dH_dp, dt)

        # gradient for Omega_k
        grad_Omega[k] = compute_gradient_single_step(
            k, F, B, dU_dO, U_target, overlap, qubit_indices
        )

        # gradient for phi_k
        grad_phi[k] = compute_gradient_single_step(
            k, F, B, dU_dp, U_target, overlap, qubit_indices
        )

    return grad_Omega, grad_phi, F_val


def numerical_gradient(Omega_list, phi_list, build_H_func, dt,
                        U_target, qubit_indices, H_kwargs=None, eps=1e-6):
    """
    Compute gradients numerically via finite differences.

    Used ONLY for testing the analytical gradients above.
    This is too slow for actual optimization (O(N) simulations
    per gradient instead of O(1)), but it is exact and serves
    as a ground truth.

    For each parameter u_k:
        dL/du_k ≈ (L(u_k + eps) - L(u_k - eps)) / (2 * eps)

    Parameters
    ----------
    (same as compute_gradients)
    eps : float
        Finite difference step size.

    Returns
    -------
    grad_Omega : numpy array, length N
    grad_phi : numpy array, length N
    """
    if H_kwargs is None:
        H_kwargs = {}

    N = len(Omega_list)
    grad_Omega = np.zeros(N)
    grad_phi   = np.zeros(N)

    def compute_L(O_list, p_list):
        H_list = pulse_to_H_list(O_list, p_list, build_H_func, **H_kwargs)
        _, _, U_total = compute_propagators(H_list, dt)
        return infidelity(U_total, U_target, qubit_indices)

    # perturb each Omega_k
    for k in range(N):
        O_plus  = Omega_list.copy(); O_plus[k]  += eps
        O_minus = Omega_list.copy(); O_minus[k] -= eps
        grad_Omega[k] = (compute_L(O_plus, phi_list) -
                         compute_L(O_minus, phi_list)) / (2 * eps)

    # perturb each phi_k
    for k in range(N):
        p_plus  = phi_list.copy(); p_plus[k]  += eps
        p_minus = phi_list.copy(); p_minus[k] -= eps
        grad_phi[k] = (compute_L(Omega_list, p_plus) -
                       compute_L(Omega_list, p_minus)) / (2 * eps)

    return grad_Omega, grad_phi

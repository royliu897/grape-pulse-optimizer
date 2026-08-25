"""
GRAPE gradients for the two-qubit CZ gate.

The two-qubit system evolves in three independent subspaces under
Rydberg blockade. The gradient for each subspace is computed using
the same forward/backward pass as the single-qubit case, then
combined into a total gradient.

The fidelity depends on the diagonal elements of each subspace
propagator: u00, u01, u11. The combined loss is:

    L = 1 - F(u00, u01, u11)

The gradient dL/dOmega_k is the sum of contributions from all
three subspaces.
"""

import numpy as np
from scipy.linalg import expm
from physics.systems.two_qubit_rydberg import build_H_two_qubit
from physics.two_qubit_propagator import (
    build_H_list_two_qubit, compute_two_qubit_propagators,
    extract_computational_element
)
from grape.gradients import dU_exact


def compute_two_qubit_gradients(Omega_list, phi_list, delta_r, dt,
                                  delta_m=0.0, cg_strong=0.5, cg_weak=None):
    """
    Compute GRAPE gradients for the CZ gate across all subspaces.

    For each subspace s in {00, 01, 11}:
        - Run forward/backward pass to get F_k^s, B_k^s
        - Compute dL/dOmega_k and dL/dphi_k contributions from subspace s
    Sum contributions across subspaces.

    Parameters
    ----------
    Omega_list : array, length N
    phi_list : array, length N
    delta_r : float
    dt : float
    delta_m, cg_strong, cg_weak : float

    Returns
    -------
    grad_Omega : array, length N
    grad_phi : array, length N
    F : float
        Current fidelity.
    """
    if cg_weak is None:
        cg_weak = 1.0 / (2.0 * np.sqrt(3))

    N = len(Omega_list)

    # build H at each timestep for each subspace
    H_lists = build_H_list_two_qubit(
        Omega_list, phi_list, delta_r, delta_m, cg_strong, cg_weak)

    # compute propagators and extract diagonal elements
    U_dict = compute_two_qubit_propagators(H_lists, dt)
    u00    = extract_computational_element(U_dict['00'])
    u01    = extract_computational_element(U_dict['01'])
    u11    = extract_computational_element(U_dict['11'])

    # fidelity
    d       = 4
    # achieved diagonal: [u00, u01, u01, u11]
    # target diagonal:   [1,   1,   1,  -1]
    # overlap = u00*1 + u01*1 + u01*1 + u11*(-1)
    overlap = u00 + u01 + u01 - u11
    F       = float(np.abs(overlap)**2 / d**2)

    # gradient of L = 1 - F w.r.t. overlap:
    # F = |overlap|^2 / d^2
    # dF/d_overlap* = overlap / d^2  (Wirtinger)
    # dL/d_overlap* = -overlap / d^2
    dL_doverlap_conj = -np.conj(overlap) / d**2

    # gradient contributions per subspace
    # d_overlap/d_u00 = 1, d_overlap/d_u01 = 2, d_overlap/d_u11 = -1
    # dL/d_u00* = dL/d_overlap* * d_overlap/d_u00 = dL_doverlap_conj * 1
    # etc.
    dL_du = {
        '00': dL_doverlap_conj * 1.0,
        '01': dL_doverlap_conj * 2.0,
        '11': dL_doverlap_conj * (-1.0),
    }

    # initialize gradient arrays
    grad_Omega = np.zeros(N)
    grad_phi   = np.zeros(N)

    # process each subspace
    for sub in ['00', '01', '11']:
        H_list = H_lists[sub]
        U_sub  = U_dict[sub]

        # forward propagators F_k = U_k @ ... @ U_1
        from physics.propagator import compute_U_single
        d_sub = H_list[0].shape[0]
        F_list = [np.eye(d_sub, dtype=complex)]
        for k in range(N):
            Uk = compute_U_single(H_list[k], dt)
            F_list.append(Uk @ F_list[-1])

        # backward propagators B_k = U_N @ ... @ U_{k+1}
        B_list = [np.eye(d_sub, dtype=complex)] * (N + 1)
        B_list[N] = np.eye(d_sub, dtype=complex)
        U_singles = [compute_U_single(H_list[k], dt) for k in range(N)]
        for k in range(N-1, -1, -1):
            B_list[k] = B_list[k+1] @ U_singles[k]

        # scalar gradient: d_loss / d_U_sub[0,0] = dL_du[sub]
        # The fidelity depends on U_sub only through U_sub[0,0].
        # We need dL/d_param_k = dL/d_U[0,0] * d_U[0,0]/d_param_k
        # d_U[0,0]/d_param_k via chain rule through B_k @ dU_k @ F_{k-1}:
        #   d_U[0,0]/d_param_k = [B_k @ dU_k @ F_{k-1}][0,0]

        for k in range(N):
            Omega_k = Omega_list[k]
            phi_k   = phi_list[k]

            # derivatives of H w.r.t. Omega and phi at this subspace
            dH_dO_sub = _dH_dOmega_subspace(phi_k, sub, cg_strong, cg_weak)
            dH_dp_sub = _dH_dphi_subspace(Omega_k, phi_k, sub, cg_strong, cg_weak)

            # exact matrix exponential derivatives
            dU_dO = dU_exact(H_list[k], dH_dO_sub, dt)
            dU_dp = dU_exact(H_list[k], dH_dp_sub, dt)

            # sandwich: B_{k+1} @ dU_k @ F_k gives d_U_total/d_param_k
            sandwich_O = B_list[k+1] @ dU_dO @ F_list[k]
            sandwich_p = B_list[k+1] @ dU_dp @ F_list[k]

            # we only care about the [0,0] element
            # dL/d_param = 2 * Re(dL/d_U[0,0]* * d_U[0,0]/d_param)
            # with dL_du[sub] = dL/d_U[0,0]*
            grad_Omega[k] += 2.0 * np.real(dL_du[sub] * sandwich_O[0, 0])
            grad_phi[k]   += 2.0 * np.real(dL_du[sub] * sandwich_p[0, 0])

    return grad_Omega, grad_phi, F


def _dH_dOmega_subspace(phi, subspace, cg_strong, cg_weak):
    """
    Derivative of subspace H with respect to Omega.
    Same coupling structure as build_H_two_qubit_subspace but with Omega=1.
    """
    # dH/dOmega = H_control / Omega
    # which is just the coupling matrix with e^(±i*phi) factors
    H = np.zeros((5, 5), dtype=complex)
    up   = np.exp(-1j * phi)
    down = np.exp(+1j * phi)

    if subspace == '00':
        H[0,1] = cg_strong*up;  H[1,0] = cg_strong*down
        H[0,2] = cg_weak*up;    H[2,0] = cg_weak*down
        H[0,3] = cg_strong*up;  H[3,0] = cg_strong*down
        H[0,4] = cg_weak*up;    H[4,0] = cg_weak*down
    elif subspace == '01':
        H[0,1] = cg_strong*up;  H[1,0] = cg_strong*down
        H[0,2] = cg_weak*up;    H[2,0] = cg_weak*down
        H[0,3] = cg_weak*up;    H[3,0] = cg_weak*down
        H[0,4] = cg_strong*up;  H[4,0] = cg_strong*down
    elif subspace == '11':
        H[0,1] = cg_weak*up;    H[1,0] = cg_weak*down
        H[0,2] = cg_strong*up;  H[2,0] = cg_strong*down
        H[0,3] = cg_weak*up;    H[3,0] = cg_weak*down
        H[0,4] = cg_strong*up;  H[4,0] = cg_strong*down

    return H


def _dH_dphi_subspace(Omega, phi, subspace, cg_strong, cg_weak):
    """
    Derivative of subspace H with respect to phi.
    Brings down -i factor for upper triangle, +i for lower.
    """
    H = np.zeros((5, 5), dtype=complex)
    up   = -1j * Omega * np.exp(-1j * phi)
    down = +1j * Omega * np.exp(+1j * phi)

    if subspace == '00':
        H[0,1] = cg_strong*up;  H[1,0] = cg_strong*down
        H[0,2] = cg_weak*up;    H[2,0] = cg_weak*down
        H[0,3] = cg_strong*up;  H[3,0] = cg_strong*down
        H[0,4] = cg_weak*up;    H[4,0] = cg_weak*down
    elif subspace == '01':
        H[0,1] = cg_strong*up;  H[1,0] = cg_strong*down
        H[0,2] = cg_weak*up;    H[2,0] = cg_weak*down
        H[0,3] = cg_weak*up;    H[3,0] = cg_weak*down
        H[0,4] = cg_strong*up;  H[4,0] = cg_strong*down
    elif subspace == '11':
        H[0,1] = cg_weak*up;    H[1,0] = cg_weak*down
        H[0,2] = cg_strong*up;  H[2,0] = cg_strong*down
        H[0,3] = cg_weak*up;    H[3,0] = cg_weak*down
        H[0,4] = cg_strong*up;  H[4,0] = cg_strong*down

    return H

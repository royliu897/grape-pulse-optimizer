"""
Two-qubit propagator for the CZ gate.

The two-qubit system under Rydberg blockade evolves independently
in three 5-dimensional subspaces {|00>, |01>, |11>}.

For GRAPE we need:
    1. The propagator U in each subspace
    2. The total 4x4 propagator in the computational basis
    3. Forward/backward passes for gradient computation

The CZ gate target in the computational basis {|00>,|01>,|10>,|11>}:
    U_CZ = diag(1, 1, 1, -1)

How this maps to the subspaces:
    |00> subspace: should return U_{00}[0,0] = 1  (no net phase on |00>)
    |01> subspace: should return U_{01}[0,0] = 1  (no net phase on |01>)
    |10> = |01> by symmetry (same dynamics, atom labels swapped)
    |11> subspace: should return U_{11}[0,0] = -1 (pi phase on |11>)

GRAPE finds Omega(t), phi(t) such that all three subspace conditions
are simultaneously satisfied.
"""

import numpy as np
from scipy.linalg import expm
from physics.systems.two_qubit_rydberg import build_H_two_qubit
from physics.propagator import compute_U_single


def build_H_list_two_qubit(Omega_list, phi_list, delta_r,
                             delta_m=0.0, cg_strong=0.5, cg_weak=None):
    """
    Build the sequence of two-qubit Hamiltonians for each GRAPE timestep.

    Returns a dict of lists: {subspace: [H_0, H_1, ..., H_{N-1}]}
    """
    N = len(Omega_list)
    H_lists = {'00': [], '01': [], '11': []}

    for k in range(N):
        H_dict = build_H_two_qubit(
            Omega_list[k], phi_list[k], delta_r, delta_m, cg_strong, cg_weak)
        for sub in ['00', '01', '11']:
            H_lists[sub].append(H_dict[sub])

    return H_lists


def compute_subspace_propagator(H_list, dt):
    """
    Compute U_total for one subspace.

    U_total = U_N @ ... @ U_1
    where U_k = expm(-i * H_k * dt)

    Parameters
    ----------
    H_list : list of 5x5 complex arrays
    dt : float

    Returns
    -------
    U_total : 5x5 complex array
    """
    d = H_list[0].shape[0]
    U = np.eye(d, dtype=complex)
    for H in H_list:
        U = compute_U_single(H, dt) @ U
    return U


def compute_two_qubit_propagators(H_lists, dt):
    """
    Compute propagators in all three subspaces.

    Parameters
    ----------
    H_lists : dict {subspace: list of H matrices}
    dt : float

    Returns
    -------
    U_dict : dict {subspace: 5x5 U_total}
    """
    return {
        sub: compute_subspace_propagator(H_lists[sub], dt)
        for sub in ['00', '01', '11']
    }


def extract_computational_element(U_sub):
    """
    Extract the |comp> -> |comp> matrix element from a subspace propagator.

    The computational state is always index 0 in each subspace basis.
    U_sub[0, 0] gives the amplitude for the initial computational state
    to return to the computational state after the gate.

    Parameters
    ----------
    U_sub : 5x5 complex array

    Returns
    -------
    element : complex scalar
    """
    return U_sub[0, 0]


def two_qubit_gate_fidelity(Omega_list, phi_list, delta_r, dt,
                              delta_m=0.0, cg_strong=0.5, cg_weak=None):
    """
    Compute CZ gate fidelity from pulse parameters.

    The CZ gate requires:
        U_{00}[0,0] =  1  (|00> returns to |00> with no phase)
        U_{01}[0,0] =  1  (|01> returns to |01> with no phase)
        U_{11}[0,0] = -1  (|11> returns to |11> with pi phase)

    Fidelity = (1/4)|Tr(U_CZ^dag @ U_computed)|^2 / 4

    Parameters
    ----------
    Omega_list : array, length N
    phi_list : array, length N
    delta_r : float
    dt : float
    delta_m, cg_strong, cg_weak : float

    Returns
    -------
    F : float
        CZ gate fidelity (0 to 1).
    U_elements : dict
        {'00': complex, '01': complex, '11': complex}
        Diagonal elements of the achieved gate.
    """
    H_lists = build_H_list_two_qubit(
        Omega_list, phi_list, delta_r, delta_m, cg_strong, cg_weak)
    U_dict  = compute_two_qubit_propagators(H_lists, dt)

    # extract computational matrix elements
    u00 = extract_computational_element(U_dict['00'])
    u01 = extract_computational_element(U_dict['01'])
    u11 = extract_computational_element(U_dict['11'])

    # build 4x4 achieved gate in computational basis
    # |10> has same diagonal element as |01> by symmetry
    U_achieved = np.diag([u00, u01, u01, u11])

    # CZ target
    U_target = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

    # fidelity
    d       = 4
    overlap = np.trace(U_target.conj().T @ U_achieved)
    F       = float(np.abs(overlap)**2 / d**2)

    return F, {'00': u00, '01': u01, '11': u11}


def two_qubit_infidelity(Omega_list, phi_list, delta_r, dt,
                          delta_m=0.0, cg_strong=0.5, cg_weak=None):
    """
    Compute CZ gate infidelity = 1 - fidelity.
    Convenience wrapper for GRAPE optimizer.
    """
    F, _ = two_qubit_gate_fidelity(
        Omega_list, phi_list, delta_r, dt, delta_m, cg_strong, cg_weak)
    return 1.0 - F

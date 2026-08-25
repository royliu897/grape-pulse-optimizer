"""
Two-qubit Hamiltonian for 171Yb Rydberg CZ gate.

The two-qubit system is built from two copies of the single-qubit
six-level Hamiltonian plus a Rydberg blockade interaction term.

State space
-----------
Each atom has 6 states (same as single qubit):
    |0>, |1>, |r_{-3/2}>, |r_{-1/2}>, |r_{+1/2}>, |r_{+3/2}>

Two atoms: 6 x 6 = 36 states in principle.

But under strong Rydberg blockade: double Rydberg excitation is
forbidden. This means states where BOTH atoms are in any Rydberg
sublevel have effectively infinite energy and can be excluded.

Effective subspace per initial computational state:
    |00> -> {|00>, |0r_{-3/2}>, |0r_{+1/2}>, |r_{-3/2}0>, |r_{+1/2}0>}
    |01> -> {|01>, |0r_{-1/2}>, |0r_{+3/2}>, |r_{-3/2}1>, |r_{+1/2}1>}
    |11> -> {|11>, |1r_{-1/2}>, |1r_{+3/2}>, |r_{-1/2}1>, |r_{+3/2}1>}
    (|01> and |10> dynamics are equivalent by symmetry)

This gives three independent 5-dimensional subspaces -- exactly what
the paper describes in the Methods section.

Implementation
--------------
We implement the full blockade Hamiltonian explicitly, then the GRAPE
optimizer works in the full space. The fidelity is projected onto the
4-dimensional computational subspace {|00>, |01>, |10>, |11>} using
qubit_indices.

For the global laser (same pulse applied to both atoms simultaneously):
    H_two = H_single ⊗ I + I ⊗ H_single + H_blockade

where H_blockade shifts double-Rydberg states to infinity (blockade limit).

In the blockade limit, we can instead work in the reduced basis
explicitly -- this is what the paper does (equation 7 and Methods).
We implement the reduced-basis version for computational efficiency.

Units: 2*pi*MHz, hbar=1.
"""

import numpy as np
import yaml
import os
from physics.systems.rydberg import build_H_drift, build_H_control, load_params


# State ordering for two-qubit system (blockade subspace)
# We use the three 5-dimensional subspaces from the paper.
# For GRAPE purposes, we work in a reduced basis per subspace.

# Subspace labels (used in build_H_two_qubit_subspace)
SUBSPACE_00 = '00'   # initial state |00>
SUBSPACE_01 = '01'   # initial state |01> (same as |10> by symmetry)
SUBSPACE_11 = '11'   # initial state |11>


def build_H_two_qubit_subspace(Omega, phi, subspace, delta_r,
                                delta_m=0.0, cg_strong=0.5, cg_weak=None):
    """
    Build the two-qubit Hamiltonian in one of the three blockade subspaces.

    Under Rydberg blockade, the two-atom dynamics factorize into three
    independent 5-dimensional subspaces depending on the initial state.
    This is the approach used in the paper (Methods, eq 7).

    Each subspace has 5 states:
        computational state + 4 single-excitation Rydberg states

    Parameters
    ----------
    Omega : float
        Rabi frequency of global laser (same laser on both atoms).
    phi : float
        Laser phase.
    subspace : str
        One of '00', '01', '11'.
    delta_r : float
        Rydberg Zeeman splitting in 2pi*MHz.
    delta_m : float
        Qubit Zeeman splitting (usually 0).
    cg_strong, cg_weak : float
        Clebsch-Gordan coefficients.

    Returns
    -------
    H : 5x5 complex numpy array
        Hamiltonian in the specified blockade subspace.

    Notes
    -----
    Basis ordering for each subspace:

    |00> subspace: {|00>, |r_{-3/2}0>, |r_{+1/2}0>, |0r_{-3/2}>, |0r_{+1/2}>}
    |01> subspace: {|01>, |r_{-3/2}1>, |r_{+1/2}1>, |0r_{-1/2}>, |0r_{+3/2}>}
    |11> subspace: {|11>, |r_{-1/2}1>, |r_{+3/2}1>, |1r_{-1/2}>, |1r_{+3/2}>}

    The laser drives transitions for EACH atom independently.
    |0> couples to |r_{-3/2}> and |r_{+1/2}> (weak couplings).
    |1> couples to |r_{-1/2}> and |r_{+3/2}> (target transition).
    """
    if cg_weak is None:
        cg_weak = 1.0 / (2.0 * np.sqrt(3))

    H = np.zeros((5, 5), dtype=complex)

    up   = Omega * np.exp(-1j * phi)
    down = Omega * np.exp(+1j * phi)

    if subspace == '00':
        # States: [|00>, |r_{-3/2}0>, |r_{+1/2}0>, |0r_{-3/2}>, |0r_{+1/2}>]
        # Diagonal: Rydberg energies
        # |00>: -2*delta_m ~ 0
        # |r_{-3/2}0>: -3*delta_r + 0 = -3*delta_r  (atom A in r_{-3/2}, B in |0>)
        # |r_{+1/2}0>: -1*delta_r
        # |0r_{-3/2}>: 0 + (-3*delta_r) = -3*delta_r  (A in |0>, B in r_{-3/2})
        # |0r_{+1/2}>: -1*delta_r
        H[0, 0] = -2 * delta_m
        H[1, 1] = -3 * delta_r   # r_{-3/2} on atom A
        H[2, 2] = -1 * delta_r   # r_{+1/2} on atom A
        H[3, 3] = -3 * delta_r   # r_{-3/2} on atom B
        H[4, 4] = -1 * delta_r   # r_{+1/2} on atom B

        # Couplings: laser drives atom A: |00> <-> |r_{-3/2}0>, |r_{+1/2}0>
        H[0, 1] = cg_strong * up;  H[1, 0] = cg_strong * down
        H[0, 2] = cg_weak   * up;  H[2, 0] = cg_weak   * down

        # Laser drives atom B: |00> <-> |0r_{-3/2}>, |0r_{+1/2}>
        H[0, 3] = cg_strong * up;  H[3, 0] = cg_strong * down
        H[0, 4] = cg_weak   * up;  H[4, 0] = cg_weak   * down

    elif subspace == '01':
        # States: [|01>, |r_{-3/2}1>, |r_{+1/2}1>, |0r_{-1/2}>, |0r_{+3/2}>]
        # Atom A starts in |0>, atom B starts in |1>
        H[0, 0] = -delta_m - delta_m   # ~ 0
        H[1, 1] = -3 * delta_r         # r_{-3/2} on A, |1> on B
        H[2, 2] = -1 * delta_r         # r_{+1/2} on A, |1> on B
        H[3, 3] = -2 * delta_r         # |0> on A, r_{-1/2} on B
        H[4, 4] = 0.0                  # |0> on A, r_{+3/2} on B (reference)

        # Laser on atom A (in |0>): |01> <-> |r_{-3/2}1>, |r_{+1/2}1>
        H[0, 1] = cg_strong * up;  H[1, 0] = cg_strong * down
        H[0, 2] = cg_weak   * up;  H[2, 0] = cg_weak   * down

        # Laser on atom B (in |1>): |01> <-> |0r_{-1/2}>, |0r_{+3/2}>
        H[0, 3] = cg_weak   * up;  H[3, 0] = cg_weak   * down
        H[0, 4] = cg_strong * up;  H[4, 0] = cg_strong * down

    elif subspace == '11':
        # States: [|11>, |r_{-1/2}1>, |r_{+3/2}1>, |1r_{-1/2}>, |1r_{+3/2}>]
        # Both atoms start in |1>
        H[0, 0] = 0.0              # both in |1> (reference)
        H[1, 1] = -2 * delta_r    # r_{-1/2} on A, |1> on B
        H[2, 2] = 0.0             # r_{+3/2} on A (TARGET), |1> on B
        H[3, 3] = -2 * delta_r    # |1> on A, r_{-1/2} on B
        H[4, 4] = 0.0             # |1> on A, r_{+3/2} on B (TARGET)

        # Laser on atom A (in |1>): |11> <-> |r_{-1/2}1>, |r_{+3/2}1>
        H[0, 1] = cg_weak   * up;  H[1, 0] = cg_weak   * down
        H[0, 2] = cg_strong * up;  H[2, 0] = cg_strong * down

        # Laser on atom B (in |1>): |11> <-> |1r_{-1/2}>, |1r_{+3/2}>
        H[0, 3] = cg_weak   * up;  H[3, 0] = cg_weak   * down
        H[0, 4] = cg_strong * up;  H[4, 0] = cg_strong * down

    else:
        raise ValueError(f"Unknown subspace: {subspace}. Use '00', '01', or '11'.")

    return H


def build_H_two_qubit(Omega, phi, delta_r, delta_m=0.0,
                       cg_strong=0.5, cg_weak=None):
    """
    Build all three subspace Hamiltonians for a global laser pulse.

    Returns a dict of {subspace: H_matrix} for use in the two-qubit
    propagator. GRAPE optimizes Omega(t) and phi(t) such that the
    combined evolution across all three subspaces implements the CZ gate.

    Parameters
    ----------
    Omega : float
        Global Rabi frequency.
    phi : float
        Laser phase.
    delta_r, delta_m, cg_strong, cg_weak : float
        Physical parameters.

    Returns
    -------
    H_dict : dict
        {'00': H_00, '01': H_01, '11': H_11}
        Each H is a 5x5 complex array.
    """
    if cg_weak is None:
        cg_weak = 1.0 / (2.0 * np.sqrt(3))

    return {
        sub: build_H_two_qubit_subspace(
            Omega, phi, sub, delta_r, delta_m, cg_strong, cg_weak)
        for sub in ['00', '01', '11']
    }


def build_H_two_qubit_from_config(Omega, phi, config_path=None):
    """
    Convenience wrapper loading parameters from config file.

    Parameters
    ----------
    Omega : float
    phi : float
    config_path : str, optional

    Returns
    -------
    H_dict : dict of 5x5 complex arrays per subspace
    """
    params = load_params(config_path)
    return build_H_two_qubit(
        Omega     = Omega,
        phi       = phi,
        delta_r   = params['delta_r_MHz'],
        delta_m   = params['delta_m_MHz'],
        cg_strong = params['cg_strong'],
        cg_weak   = params['cg_weak'],
    )

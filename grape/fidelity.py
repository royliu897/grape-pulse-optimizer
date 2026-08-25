"""
Fidelity module.

Computes the gate fidelity between the achieved unitary U
and the target unitary U_target.

The fidelity measure used is:

    F = (1/d^2) * |Tr(U_target† @ U)|^2

where d is the dimension of the target gate (2 for single qubit,
4 for two qubit). This ranges from 0 (completely wrong) to 1 (perfect).

GRAPE minimizes the infidelity:

    L = 1 - F

Key subtlety -- subspace projection:
    Our physical system has more states than the gate acts on.
    For example, the 6-level Rydberg system has 2 qubit states
    and 4 Rydberg states, but the gate target is only 2x2.
    We project U_total down to just the qubit subspace before
    computing fidelity.
"""

import numpy as np


def project_to_subspace(U, qubit_indices):
    """
    Project a unitary down to the qubit subspace.

    Takes the rows and columns corresponding to the qubit states,
    giving us only the part of U that acts on the computational space.

    Parameters
    ----------
    U : numpy array, shape (d_full, d_full)
        Full unitary in the physical Hilbert space.
        For the 6-level Rydberg system, d_full = 6.
    qubit_indices : list of int
        Indices of the qubit states in the full Hilbert space.
        For the Rydberg system: [0, 1] (|0> and |1> are indices 0 and 1).

    Returns
    -------
    U_sub : numpy array, shape (d_sub, d_sub)
        The projected unitary acting only on the qubit subspace.
        For the Rydberg system: 2x2 matrix.

    Notes
    -----
    This is just fancy indexing -- we extract the submatrix formed
    by the rows and columns at qubit_indices.

    Example: if qubit_indices = [0, 1] and U is 6x6:
        U_sub = U[[0,1], :][:, [0,1]]
        which is just the top-left 2x2 block.

    Physical meaning: U_sub[i,j] is the amplitude for the system
    to go from qubit state j to qubit state i, given that it started
    and ended in the qubit subspace (no leakage to Rydberg states).
    """
    idx = np.array(qubit_indices)
    return U[np.ix_(idx, idx)]


def gate_fidelity(U, U_target, qubit_indices=None):
    """
    Compute the gate fidelity between U and U_target.

    F = (1/d^2) * |Tr(U_target† @ U_sub)|^2

    Parameters
    ----------
    U : numpy array
        Achieved unitary. Can be full physical unitary (e.g. 6x6)
        or already projected to the qubit subspace.
    U_target : numpy array, shape (d, d)
        Target gate unitary. Always in the qubit subspace.
        Examples:
            X gate:  [[0, 1], [1, 0]]
            CZ gate: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]]
    qubit_indices : list of int, optional
        If provided, projects U down to this subspace first.
        If None, assumes U is already in the right subspace.

    Returns
    -------
    F : float
        Gate fidelity, between 0 and 1.

    Notes
    -----
    The |...|^2 makes fidelity insensitive to global phase.
    A global phase e^(i*theta) on U gives the same fidelity as U.
    This is physically correct -- global phases are unobservable.
    """
    # project if needed
    if qubit_indices is not None:
        U_sub = project_to_subspace(U, qubit_indices)
    else:
        U_sub = U

    d = U_target.shape[0]

    # overlap = Tr(U_target† @ U_sub)
    # this is a complex number measuring how much U_sub looks like U_target
    overlap = np.trace(U_target.conj().T @ U_sub)

    # fidelity = |overlap|^2 / d^2
    F = (np.abs(overlap) ** 2) / (d ** 2)

    return float(F)


def infidelity(U, U_target, qubit_indices=None):
    """
    Compute the gate infidelity: L = 1 - F.

    This is what GRAPE minimizes. Zero means perfect gate.

    Parameters
    ----------
    U : numpy array
        Achieved unitary.
    U_target : numpy array
        Target gate unitary.
    qubit_indices : list of int, optional
        Subspace projection indices.

    Returns
    -------
    L : float
        Infidelity, between 0 and 1.
    """
    return 1.0 - gate_fidelity(U, U_target, qubit_indices)


def fidelity_and_overlap(U, U_target, qubit_indices=None):
    """
    Compute both the fidelity and the raw complex overlap.

    The raw overlap Tr(U_target† @ U_sub) is needed by the
    gradient computation, so we expose it here to avoid
    computing it twice.

    Parameters
    ----------
    U : numpy array
        Achieved unitary.
    U_target : numpy array
        Target gate unitary.
    qubit_indices : list of int, optional
        Subspace projection indices.

    Returns
    -------
    F : float
        Gate fidelity.
    overlap : complex
        Raw complex overlap Tr(U_target† @ U_sub).
        Needed for gradient computation.
    U_sub : numpy array
        The projected unitary (or U itself if no projection).
        Also needed for gradient computation.
    """
    if qubit_indices is not None:
        U_sub = project_to_subspace(U, qubit_indices)
    else:
        U_sub = U

    d = U_target.shape[0]
    overlap = np.trace(U_target.conj().T @ U_sub)
    F = float((np.abs(overlap) ** 2) / (d ** 2))

    return F, overlap, U_sub


# --- Standard gate targets ---
# These are the U_target matrices for common gates.
# Stored here for convenience so you don't have to look them up.

def target_X():
    """Single qubit X (NOT) gate."""
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def target_Y():
    """Single qubit Y gate."""
    return np.array([[0, -1j],
                     [1j,  0]], dtype=complex)

def target_Z():
    """Single qubit Z gate."""
    return np.array([[1,  0],
                     [0, -1]], dtype=complex)

def target_Hadamard():
    """Single qubit Hadamard gate."""
    return np.array([[1,  1],
                     [1, -1]], dtype=complex) / np.sqrt(2)

def target_CZ():
    """
    Two qubit controlled-Z gate.

    Acts on the 4-dimensional two-qubit space {|00>, |01>, |10>, |11>}.
    Applies a phase of -1 to |11> only.

        |00> ->  |00>
        |01> ->  |01>
        |10> ->  |10>
        |11> -> -|11>
    """
    return np.diag([1, 1, 1, -1]).astype(complex)

def target_pi_pulse():
    """
    A pi pulse on the single qubit -- same as X gate.
    Named separately because in the Rydberg context this often
    refers specifically to the |1> -> |r> -> |1> round trip.
    """
    return target_X()

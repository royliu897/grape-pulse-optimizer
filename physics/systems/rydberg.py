"""
Hamiltonian for the metastable 171Yb Rydberg qubit.

This implements equation (6) from Ma et al. 2023.

The system has 6 states, in this exact order:
    index 0 : |0>        = |3P0, mF = -1/2>  (qubit spin down)
    index 1 : |1>        = |3P0, mF = +1/2>  (qubit spin up)
    index 2 : |r_{-3/2}> = Rydberg mF = -3/2
    index 3 : |r_{-1/2}> = Rydberg mF = -1/2
    index 4 : |r_{+1/2}> = Rydberg mF = +1/2
    index 5 : |r_{+3/2}> = Rydberg mF = +3/2  <-- target state

The Hamiltonian has two parts:
    H(t) = H_drift + H_control(Omega, phi)

H_drift  : fixed atomic physics (Zeeman splitting of Rydberg levels)
H_control: laser coupling, changes at each GRAPE timestep

Units: everything in 2*pi*MHz, hbar set to 1
"""

import numpy as np
import yaml
import os


def load_params(config_path=None):
    """Load physical parameters from the yaml config file."""
    if config_path is None:
        # default: look for config relative to this file's location
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, '../../config/rydberg_params.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_H_drift(delta_r, delta_m=0.0):
    """
    Build the drift Hamiltonian -- the part that is always present
    regardless of what the laser is doing.

    This is purely the diagonal part of eq (6) -- the Zeeman energies
    of each state. The laser is off (Omega = 0), so all off-diagonal
    entries are zero.

    Parameters
    ----------
    delta_r : float
        Zeeman splitting between adjacent Rydberg sublevels, in 2pi*MHz.
        From the paper: delta_r = 2pi * 9.3 MHz at B = 5 Gauss.
    delta_m : float
        Zeeman splitting of the qubit states in 3P0.
        Effectively 0 in this system -- included for completeness.

    Returns
    -------
    H_drift : 6x6 complex numpy array

    Notes on the diagonal values (reading eq 6 top to bottom):
        |0>        : -delta_m  (tiny, ~0)
        |1>        : 0         (reference energy, laser tuned resonant here)
        |r_{-3/2}> : -3*delta_r
        |r_{-1/2}> : -2*delta_r
        |r_{+1/2}> : -1*delta_r
        |r_{+3/2}> : 0         (reference Rydberg energy)
    """
    H = np.zeros((6, 6), dtype=complex)

    # diagonal entries = energy of each state
    H[0, 0] = -delta_m      # |0>  qubit spin down
    H[1, 1] = 0.0           # |1>  qubit spin up (laser resonant with this)
    H[2, 2] = -3 * delta_r  # |r_{-3/2}>
    H[3, 3] = -2 * delta_r  # |r_{-1/2}>
    H[4, 4] = -1 * delta_r  # |r_{+1/2}>
    H[5, 5] = 0.0           # |r_{+3/2}>  target Rydberg state

    return H


def build_H_control(Omega, phi, cg_strong=0.5, cg_weak=None):
    """
    Build the control Hamiltonian -- the part driven by the laser.

    This is the off-diagonal part of eq (6). It is zero when Omega=0
    (laser off) and nonzero when the laser is on.

    Omega and phi are the two knobs GRAPE optimizes.

    Parameters
    ----------
    Omega : float
        Laser Rabi frequency at this timestep, in 2pi*MHz.
        Controls the amplitude/strength of the coupling.
    phi : float
        Laser phase at this timestep, in radians.
        Controls the direction of the coupling in the complex plane.
    cg_strong : float
        Clebsch-Gordan coefficient for the strong transitions (1/2).
    cg_weak : float
        Clebsch-Gordan coefficient for the weak transitions (1/2*sqrt(3)).
        If None, computed automatically.

    Returns
    -------
    H_control : 6x6 complex numpy array

    Notes on which states couple to which (reading eq 6):
        |0> (row 0) couples to:
            |r_{-3/2}> (col 2) with strength cg_strong  -- Delta_mF = -1 (sigma-)
            |r_{+1/2}> (col 4) with strength cg_weak    -- Delta_mF = +1 (sigma+)
            NOT to |r_{+3/2}> -- forbidden, Delta_mF would need to be +2

        |1> (row 1) couples to:
            |r_{-1/2}> (col 3) with strength cg_weak    -- off-resonant, unwanted
            |r_{+3/2}> (col 5) with strength cg_strong  -- TARGET transition

    The e^(+i*phi) / e^(-i*phi) conjugate pairs ensure H is Hermitian.
    Upper triangle has e^(-i*phi), lower triangle has e^(+i*phi).
    """
    if cg_weak is None:
        cg_weak = 1.0 / (2.0 * np.sqrt(3))

    H = np.zeros((6, 6), dtype=complex)

    # complex coupling amplitude -- this is what Omega and phi control
    # upper triangle uses e^(-i*phi), lower uses e^(+i*phi) (Hermitian conjugate)
    up   = Omega * np.exp(-1j * phi)   # drives upward transitions
    down = Omega * np.exp(+1j * phi)   # conjugate, drives downward (Hermitian pair)

    # --- couplings FROM |0> (row 0, col >0) ---
    # |0> -> |r_{-3/2}>  (index 0 -> index 2)
    H[0, 2] = cg_strong * up
    H[2, 0] = cg_strong * down   # Hermitian conjugate

    # |0> -> |r_{+1/2}>  (index 0 -> index 4)
    H[0, 4] = cg_weak * up
    H[4, 0] = cg_weak * down     # Hermitian conjugate

    # NOTE: H[0, 5] = 0  -- |0> cannot reach |r_{+3/2}>, hard forbidden

    # --- couplings FROM |1> (row 1, col >1) ---
    # |1> -> |r_{-1/2}>  (index 1 -> index 3)  -- unwanted off-resonant coupling
    H[1, 3] = cg_weak * up
    H[3, 1] = cg_weak * down     # Hermitian conjugate

    # |1> -> |r_{+3/2}>  (index 1 -> index 5)  -- TARGET transition
    H[1, 5] = cg_strong * up
    H[5, 1] = cg_strong * down   # Hermitian conjugate

    return H


def build_H(Omega, phi, delta_r, delta_m=0.0, cg_strong=0.5, cg_weak=None):
    """
    Build the full Hamiltonian at a single timestep.

    H(t) = H_drift + H_control(Omega, phi)

    This is what gets called at every GRAPE timestep with the current
    pulse parameters.

    Parameters
    ----------
    Omega : float
        Laser Rabi frequency at this timestep.
    phi : float
        Laser phase at this timestep.
    delta_r : float
        Rydberg Zeeman splitting (fixed physical parameter).
    delta_m : float
        Qubit Zeeman splitting (effectively 0).
    cg_strong, cg_weak : float
        Clebsch-Gordan coefficients.

    Returns
    -------
    H : 6x6 complex numpy array
    """
    H_drift   = build_H_drift(delta_r, delta_m)
    H_control = build_H_control(Omega, phi, cg_strong, cg_weak)
    return H_drift + H_control


def build_H_from_config(Omega, phi, config_path=None):
    """
    Convenience function: load parameters from config and build H.
    This is what most code will call in practice.

    Parameters
    ----------
    Omega : float
        Laser Rabi frequency at this timestep.
    phi : float
        Laser phase at this timestep.
    config_path : str, optional
        Path to yaml config. Uses default if None.

    Returns
    -------
    H : 6x6 complex numpy array
    """
    params = load_params(config_path)
    return build_H(
        Omega    = Omega,
        phi      = phi,
        delta_r  = params['delta_r_MHz'],
        delta_m  = params['delta_m_MHz'],
        cg_strong= params['cg_strong'],
        cg_weak  = params['cg_weak'],
    )

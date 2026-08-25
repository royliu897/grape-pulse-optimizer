"""
Regularization terms for GRAPE pulse optimization.

Adds penalty terms to the infidelity loss to produce hardware-friendly
pulses that the AOM can actually reproduce.

Total loss:
    L_total = L_fidelity + alpha * L_power + beta * L_smoothness + gamma * L_envelope

Physical interpretation of each term
-------------------------------------
L_power (alpha):
    sum(Omega_k^2) * dt
    Penalizes total pulse energy. Prevents unrealistic high-power solutions.
    Keep small (alpha ~ 1e-4) -- mainly a safety constraint.

L_smoothness (beta):
    sum((Omega_{k+1} - Omega_k)^2 / dt) + sum((phi_{k+1} - phi_k)^2 / dt)
    Penalizes rapid changes in amplitude AND phase.
    This is the primary regularizer -- directly encodes bandwidth constraints.
    Physical connection: beta ~ safety_factor / (2*pi*bandwidth_MHz)^2
    For 10 MHz AOM: beta_physical ~ 2.5e-4. Use beta=0.01 as safe starting point.
    Increase if hardware correction still fails. Decrease if fidelity suffers.

L_envelope (gamma):
    sum((Omega_k - Omega_gaussian_k)^2) * dt
    Softly encourages Gaussian amplitude shape.
    The paper (Ma et al. 2023) actually fixes Omega(t) to have Gaussian
    rising/falling edges and only optimizes phi(t). This term mimics that
    without hard-constraining the amplitude.
    gamma=0 means off (default). gamma~0.1 gives noticeable shaping.

Tuning guide
------------
Start with: alpha=0, beta=0.01, gamma=0
If hardware correction still fails (phase jumps): increase beta to 0.05
If fidelity drops too much: decrease beta or increase N_timesteps
If pulse looks physically unreasonable: add small gamma ~ 0.05
"""

import numpy as np


def regularization_loss(Omega_list, phi_list, dt,
                         alpha=0.0, beta=0.01, gamma=0.0,
                         Omega_target=None):
    """
    Compute regularization penalty and gradients.

    Parameters
    ----------
    Omega_list : array, length N
    phi_list   : array, length N
    dt         : float, timestep in microseconds
    alpha      : float, power penalty weight
    beta       : float, smoothness penalty weight (amplitude + phase)
    gamma      : float, Gaussian envelope penalty weight
    Omega_target : array length N or None
        Target Gaussian envelope for gamma penalty.
        Auto-generated from mean amplitude if None and gamma > 0.

    Returns
    -------
    L_reg      : float, total regularization loss
    grad_Omega : array length N, d_L_reg/d_Omega_k
    grad_phi   : array length N, d_L_reg/d_phi_k
    breakdown  : dict, individual term values for diagnostics
    """
    N          = len(Omega_list)
    grad_Omega = np.zeros(N)
    grad_phi   = np.zeros(N)
    breakdown  = {}

    # --- Power penalty: alpha * sum(Omega_k^2) * dt ---
    if alpha > 0.0:
        L_power        = alpha * np.sum(Omega_list**2) * dt
        grad_Omega    += 2.0 * alpha * Omega_list * dt
        breakdown['L_power'] = float(L_power)
    else:
        breakdown['L_power'] = 0.0

    # --- Smoothness penalty ---
    # Integral approximation of alpha*||dOmega/dt||^2 + beta*||dphi/dt||^2
    # Gradient is the discrete Laplacian (second finite difference).
    # For interior k: d/dOmega_k sum_{j}(Omega_{j+1}-Omega_j)^2/dt
    #   = 2*(2*Omega_k - Omega_{k-1} - Omega_{k+1}) / dt
    # Boundary terms only have one neighbor.
    if beta > 0.0:
        # amplitude smoothness
        dO             = np.diff(Omega_list)            # length N-1
        L_smooth_O     = beta * np.sum(dO**2) / dt
        g_O            = np.zeros(N)
        g_O[0]         =  2.0 * beta * (Omega_list[0]  - Omega_list[1])  / dt
        g_O[-1]        =  2.0 * beta * (Omega_list[-1] - Omega_list[-2]) / dt
        g_O[1:-1]      =  2.0 * beta * (2*Omega_list[1:-1]
                                         - Omega_list[:-2]
                                         - Omega_list[2:]) / dt
        grad_Omega    += g_O

        # phase smoothness
        dp             = np.diff(phi_list)
        L_smooth_p     = beta * np.sum(dp**2) / dt
        g_p            = np.zeros(N)
        g_p[0]         =  2.0 * beta * (phi_list[0]  - phi_list[1])  / dt
        g_p[-1]        =  2.0 * beta * (phi_list[-1] - phi_list[-2]) / dt
        g_p[1:-1]      =  2.0 * beta * (2*phi_list[1:-1]
                                         - phi_list[:-2]
                                         - phi_list[2:]) / dt
        grad_phi      += g_p

        breakdown['L_smooth_amplitude'] = float(L_smooth_O)
        breakdown['L_smooth_phase']     = float(L_smooth_p)
    else:
        breakdown['L_smooth_amplitude'] = 0.0
        breakdown['L_smooth_phase']     = 0.0

    # --- Gaussian envelope penalty: gamma * sum((Omega_k - Omega_gauss_k)^2)*dt ---
    # Inspired by the paper which uses Gaussian rising/falling edges for Omega(t).
    if gamma > 0.0:
        if Omega_target is None:
            t       = np.arange(N) * dt + dt / 2.0
            t_mid   = t[N // 2]
            sigma   = (t[-1] - t[0]) / 4.0
            A_peak  = np.mean(Omega_list)
            Omega_target = A_peak * np.exp(-((t - t_mid)**2) / (2.0 * sigma**2))

        diff           = Omega_list - Omega_target
        L_env          = gamma * np.sum(diff**2) * dt
        grad_Omega    += 2.0 * gamma * diff * dt
        breakdown['L_envelope'] = float(L_env)
    else:
        breakdown['L_envelope'] = 0.0

    L_reg = sum(breakdown.values())
    return L_reg, grad_Omega, grad_phi, breakdown


def suggest_beta(bandwidth_MHz, safety_factor=10.0):
    """
    Suggest a smoothness penalty beta from AOM bandwidth.

        beta = safety_factor / (2*pi*B)^2

    With safety_factor=10, we penalize frequency components at 10x
    below the hard bandwidth limit -- conservative but safe.

    Parameters
    ----------
    bandwidth_MHz : float
    safety_factor : float

    Returns
    -------
    beta : float
    """
    omega_cutoff = 2.0 * np.pi * bandwidth_MHz   # rad/us
    return safety_factor / omega_cutoff**2

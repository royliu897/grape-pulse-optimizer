"""
Hardware pipeline for modeling optical pulse distortion.

Architecture
------------
The pipeline is a composed differentiable operator:

    F_total = F_N ∘ ... ∘ F_2 ∘ F_1

Correction is a unified optimization problem:

    find input such that F_total(input) ≈ target

solved by gradient descent (Adam) through the full pipeline via
backpropagation. No component is inverted independently.

Key design decisions:
    1. All signals in complex envelope space A(t) -- never extract phase.
       This avoids np.unwrap (non-differentiable discontinuities).

    2. Each component exposes forward() and vjp() -- the pipeline
       chains them via the chain rule, exactly like GRAPE's adjoint method.

    3. Adam optimizer for correction -- adaptive learning rates per
       parameter, robust to the varying gradient magnitudes across
       the pulse (edges vs middle).

    4. Modular -- add/remove components freely, correction adapts
       automatically because it optimizes through the full pipeline.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod
from scipy.fft import fft, ifft, fftfreq


# ---------------------------------------------------------------------------
# Pulse dataclass -- complex envelope representation
# ---------------------------------------------------------------------------

@dataclass
class Pulse:
    """
    Laser pulse as a complex envelope sampled in time.

        A(t) = Omega(t) * exp(i * phi(t))

    We store only A -- never extract or store Omega and phi separately.
    This keeps everything differentiable.

    Parameters
    ----------
    t : numpy array
        Time axis in microseconds.
    A : numpy array, complex
        Complex envelope.
    """
    t : np.ndarray
    A : np.ndarray

    def __post_init__(self):
        self.A = np.array(self.A, dtype=complex)
        assert len(self.t) == len(self.A)

    @property
    def N(self):
        return len(self.t)

    @property
    def dt(self):
        return self.t[1] - self.t[0]

    @property
    def Omega(self):
        """Amplitude |A(t)|. Read-only derived property."""
        return np.abs(self.A)

    @property
    def phi(self):
        """Unwrapped phase. Use only for display/plotting, not in gradients."""
        return np.unwrap(np.angle(self.A))

    def copy(self):
        return Pulse(t=self.t.copy(), A=self.A.copy())

    @classmethod
    def from_polar(cls, t, Omega, phi):
        """Construct from amplitude and phase arrays."""
        return cls(t=t, A=Omega * np.exp(1j * phi))

    @classmethod
    def from_grape(cls, Omega_list, phi_list, duration_us):
        """Construct from GRAPE output."""
        N  = len(Omega_list)
        dt = duration_us / N
        t  = np.arange(N) * dt + dt / 2
        return cls.from_polar(t, np.array(Omega_list), np.array(phi_list))

    def to_grape(self):
        """Extract GRAPE-compatible arrays. Uses unwrap -- for display only."""
        return self.Omega.copy(), self.phi.copy()

    def loss_vs(self, target: 'Pulse') -> float:
        """MSE loss in complex envelope space."""
        return float(np.mean(np.abs(self.A - target.A)**2))

    def grad_of_loss_vs(self, target: 'Pulse') -> np.ndarray:
        """
        Gradient of MSE loss w.r.t. A*.
        d/dA* ||A - A_target||^2 / N = (A - A_target) / N
        """
        return (self.A - target.A) / self.N


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class HardwareComponent(ABC):
    """
    Abstract base class for a hardware component.

    Subclasses must implement:

        forward(pulse) -> pulse
            The physical distortion model.

        vjp(pulse_in, grad_out) -> grad_in
            Vector-Jacobian product (backpropagation through this component).
            grad_out: d_loss/d_A_out (complex array)
            grad_in:  d_loss/d_A_in  (complex array)

            Override with analytical VJP. Default is numerical finite
            differences -- always correct but O(N^2) and slow.
    """

    @abstractmethod
    def forward(self, pulse: Pulse) -> Pulse:
        pass

    def vjp(self, pulse_in: Pulse, grad_out: np.ndarray) -> np.ndarray:
        """
        Default VJP via finite differences.
        For verification of analytical VJPs only -- too slow for production.
        """
        eps     = 1e-7
        A_in    = pulse_in.A.copy()
        N       = len(A_in)
        grad_in = np.zeros(N, dtype=complex)
        A0_out  = self.forward(pulse_in).A

        for k in range(N):
            A_r         = A_in.copy(); A_r[k] += eps
            dA_out_dr   = (self.forward(Pulse(pulse_in.t, A_r)).A - A0_out) / eps
            A_i         = A_in.copy(); A_i[k] += 1j * eps
            dA_out_di   = (self.forward(Pulse(pulse_in.t, A_i)).A - A0_out) / eps

            grad_in[k] = (
                np.dot(np.real(grad_out), np.real(dA_out_dr))
              + np.dot(np.imag(grad_out), np.imag(dA_out_dr))
              + 1j * np.dot(np.real(grad_out), np.real(dA_out_di))
              + 1j * np.dot(np.imag(grad_out), np.imag(dA_out_di))
            )

        return grad_in

    def parameter_summary(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Hardware Pipeline
# ---------------------------------------------------------------------------

class HardwarePipeline:
    """
    Differentiable chain of hardware components.

    Correction uses Adam optimizer through the full pipeline via
    backpropagation. Swapping components requires no changes to
    the correction logic.
    """

    def __init__(self, components: List[HardwareComponent]):
        self.components = components

    def forward(self, pulse: Pulse) -> Pulse:
        """Pass pulse through all components in sequence."""
        current = pulse.copy()
        for component in self.components:
            current = component.forward(current)
        return current

    def _forward_with_intermediates(self, pulse: Pulse):
        """Forward pass storing intermediates for backprop."""
        intermediates = [pulse.copy()]
        current = pulse.copy()
        for component in self.components:
            current = component.forward(current)
            intermediates.append(current.copy())
        return intermediates

    def backpropagate(self, pulse_in: Pulse,
                      grad_output: np.ndarray) -> np.ndarray:
        """
        Chain rule through all components in reverse order.

        grad_output : d_loss/d_A_output
        returns     : d_loss/d_A_input
        """
        intermediates = self._forward_with_intermediates(pulse_in)
        grad = grad_output.copy()
        for k in range(len(self.components) - 1, -1, -1):
            grad = self.components[k].vjp(intermediates[k], grad)
        return grad

    def correct(self, target: Pulse,
                n_iter: int = 300,
                learning_rate: float = 0.05,
                beta1: float = 0.9,
                beta2: float = 0.999,
                epsilon: float = 1e-8,
                verbose: bool = False) -> Tuple[Pulse, List[float]]:
        """
        Find input pulse whose output matches target.

        Uses Adam optimizer -- adaptive per-parameter learning rates,
        robust to varying gradient magnitudes across the pulse.

        Adam update rule:
            m = beta1*m + (1-beta1)*grad          (first moment / mean)
            v = beta2*v + (1-beta2)*grad^2        (second moment / variance)
            m_hat = m / (1-beta1^t)               (bias correction)
            v_hat = v / (1-beta2^t)
            theta -= lr * m_hat / (sqrt(v_hat) + eps)

        Why Adam over SGD+momentum:
            - Adapts lr per parameter (edges of pulse have larger gradients)
            - Bias correction prevents large early steps
            - More robust to curvature differences across the landscape

        Parameters
        ----------
        target : Pulse
            Desired output (what atoms see).
        n_iter : int
            Adam iterations.
        learning_rate : float
            Global step size (Adam adapts per-parameter on top of this).
        beta1, beta2 : float
            Adam moment decay rates.
        epsilon : float
            Adam numerical stability constant.
        verbose : bool
            Print loss every 50 iters.

        Returns
        -------
        corrected_input : Pulse
        loss_history : list of float
        """
        # warm start from target
        current_A = target.A.copy()

        # Adam state (complex -- real and imag parts treated independently)
        m = np.zeros_like(current_A)   # first moment
        v = np.zeros(len(current_A))   # second moment (real, tracks magnitude^2)
        loss_history = []

        for i in range(1, n_iter + 1):
            pulse_in  = Pulse(target.t, current_A)
            pulse_out = self.forward(pulse_in)
            loss      = pulse_out.loss_vs(target)
            loss_history.append(loss)

            if verbose and i % 50 == 0:
                print(f"  iter {i:4d}  |  loss = {loss:.6e}")

            if loss < 1e-12:
                break

            # gradient of loss w.r.t. A_input
            grad_out = pulse_out.grad_of_loss_vs(target)
            grad_in  = self.backpropagate(pulse_in, grad_out)

            # Adam update (treating complex gradient as 2D real)
            m = beta1 * m + (1 - beta1) * grad_in
            v = beta2 * v + (1 - beta2) * np.abs(grad_in)**2

            # bias correction
            m_hat = m / (1 - beta1**i)
            v_hat = v / (1 - beta2**i)

            # parameter update
            current_A = current_A - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        return Pulse(target.t, current_A), loss_history

    def add_component(self, component: HardwareComponent, index: int = None):
        if index is None:
            self.components.append(component)
        else:
            self.components.insert(index, component)

    def remove_component(self, index: int):
        self.components.pop(index)

    def summary(self):
        print(f"HardwarePipeline: {len(self.components)} components")
        for i, comp in enumerate(self.components):
            print(f"  [{i}] {comp.__class__.__name__}")
            for k, v in comp.parameter_summary().items():
                print(f"       {k}: {v}")

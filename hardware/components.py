import numpy as np
from scipy.fft import fft, ifft, fftfreq
from hardware.pipeline import Pulse, HardwareComponent


class AOMModel(HardwareComponent):
    """
    Acousto-optic modulator model.

    forward() applies in order:
        1. Bandwidth limiting  -- Gaussian LPF on complex envelope
        2. Phase transients    -- state-space IIR on reconstructed phase
        3. Noise               -- stochastic, excluded from VJP

    vjp() backpropagates analytically through steps 1 and 2.

    Parameters
    ----------
    bandwidth_MHz : float
        3dB bandwidth in MHz. Typical UV AOM: 5-20 MHz.
    settling_time_us : float
        Phase settling time constant tau. alpha = exp(-dt/tau).
    phase_overshoot : float
        IIR (Infinite Impulse Response) overshoot coefficient beta (0 to ~0.5).
    amplitude_noise : float
        RMS fractional amplitude noise.
    phase_noise_rad : float
        RMS phase noise per sqrt(us).
    seed : int, optional
        Random seed for reproducible noise.
    """

    def __init__(self, bandwidth_MHz=10.0, settling_time_us=0.15,
                 phase_overshoot=0.2, amplitude_noise=0.005,
                 phase_noise_rad=0.005, seed=None):
        self.bandwidth_MHz    = bandwidth_MHz
        self.settling_time_us = settling_time_us
        self.phase_overshoot  = phase_overshoot
        self.amplitude_noise  = amplitude_noise
        self.phase_noise_rad  = phase_noise_rad
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Bandwidth filter
    # ------------------------------------------------------------------

    def _H_bw(self, N, dt):
        """Gaussian low-pass transfer function."""
        freqs = fftfreq(N, d=dt)
        return np.exp(-freqs**2 / (2.0 * self.bandwidth_MHz**2))

    def _apply_bw(self, A, N, dt):
        return ifft(fft(A) * self._H_bw(N, dt))

    # ------------------------------------------------------------------
    # Phase state-space IIR
    # ------------------------------------------------------------------

    def _iir_coeffs(self, dt):
        alpha = np.exp(-dt / self.settling_time_us)
        beta  = self.phase_overshoot
        c     = 1.0 - alpha + beta
        return alpha, beta, c

    def _reconstruct_phase(self, A_bw):
        """
        Reconstruct continuous phase from complex envelope without unwrap.

        Uses local phase differences:
            delta_phi[n] = angle(A_bw[n] * conj(A_bw[n-1]))

        This is the imaginary part of log(A[n]/A[n-1]), evaluated
        locally so it stays in (-pi, pi) as long as the phase doesn't
        jump more than pi between adjacent samples -- which is guaranteed
        by the bandwidth limit for any physical pulse.

        Returns
        -------
        phi : real array, length N
            Reconstructed phase trajectory.
        delta_phi : real array, length N
            Phase differences. delta_phi[0] = angle(A_bw[0]).
        """
        N         = len(A_bw)
        phi       = np.zeros(N)
        delta_phi = np.zeros(N)

        # first sample: use global angle (only one sample, no discontinuity issue)
        delta_phi[0] = np.angle(A_bw[0])
        phi[0]       = delta_phi[0]

        for n in range(1, N):
            # local phase difference -- bounded in (-pi, pi) by bandwidth
            delta_phi[n] = np.angle(A_bw[n] * np.conj(A_bw[n-1]))

            if np.abs(A_bw[n]) > 0.05 and np.abs(delta_phi[n]) > 0.99 * np.pi:
                raise ValueError(
                    f"Phase jump too large at index {n} (delta = {delta_phi[n]:.3f} rad). "
                    "Bandwidth filter or sampling rate (dt) may be insufficient."
                )
                
            phi[n]       = phi[n-1] + delta_phi[n]

        return phi, delta_phi

    def _apply_iir(self, phi, alpha, beta, c):
        """
        Real-valued IIR filter on phase.

        s[0] = phi[0]
        s[n] = alpha*s[n-1] + c*phi[n] - beta*phi[n-1]

        Returns the filtered phase s.
        """
        N = len(phi)
        s    = np.zeros(N)
        s[0] = phi[0]
        for n in range(1, N):
            s[n] = alpha*s[n-1] + c*phi[n] - beta*phi[n-1]
        return s

    def _vjp_iir(self, g_s, alpha, beta, c):
        """
        VJP of real IIR: given d_loss/d_s, return d_loss/d_phi.

        From the Jacobian analysis: J[n,k] = alpha^(n-k-1)*J[k+1,k] for n>k+1
        VJP[k] = c*g[k] + (alpha*c - beta)*S[k+1]
        where S[k] = g[k] + alpha*S[k+1] (backward weighted sum).

        Special case k=0: VJP[0] = 1*g[0] + (alpha-beta)*S[1]
        (because s[0]=phi[0] directly, coefficient is 1 not c)

        Verified against numerical J^T @ g to machine precision.
        """
        N = len(g_s)
        # backward weighted sum
        S    = np.zeros(N + 1)
        for k in range(N - 1, -1, -1):
            S[k] = g_s[k] + alpha * S[k + 1]

        g_phi    = np.zeros(N)
        g_phi[0] = 1.0 * g_s[0] + (alpha - beta) * S[1]
        for k in range(1, N):
            g_phi[k] = c * g_s[k] + (alpha * c - beta) * S[k + 1]

        return g_phi

    def _vjp_reconstruct_phase(self, A_bw, g_phi):
        """
        VJP of phase reconstruction step using decoupled angles.
        """
        N          = len(A_bw)
        g_A_bw     = np.zeros(N, dtype=complex)

        # reverse cumsum: g_delta_phi[k] = sum_{n>=k} g_phi[n]
        g_delta_phi        = np.zeros(N)
        g_delta_phi[N - 1] = g_phi[N - 1]
        for k in range(N - 2, -1, -1):
            g_delta_phi[k] = g_phi[k] + g_delta_phi[k + 1]

        # Since angle(A[n] * conj(A[n-1])) = angle(A[n]) - angle(A[n-1]),
        # the gradients completely decouple! d/dA angle(A) = i * A / |A|^2
        for n in range(N):
            term = 1j * A_bw[n] / (np.abs(A_bw[n])**2 + 1e-20)
            
            if n == 0:
                g_A_bw[0] += g_delta_phi[0] * term
            else:
                g_A_bw[n]   += g_delta_phi[n] * term
                g_A_bw[n-1] -= g_delta_phi[n] * 1j * A_bw[n-1] / (np.abs(A_bw[n-1])**2 + 1e-20)

        return g_A_bw

    # ------------------------------------------------------------------
    # Forward and VJP
    # ------------------------------------------------------------------

    def forward(self, pulse: Pulse) -> Pulse:
        """
        Apply AOM distortions:
            1. Bandwidth (Gaussian LPF on complex envelope)
            2. Phase transients (state-space IIR on reconstructed phase)
            3. Noise (stochastic, excluded from VJP)
        """
        N  = pulse.N
        dt = pulse.dt
        alpha, beta, c = self._iir_coeffs(dt)

        # 1. bandwidth filter
        A_bw = self._apply_bw(pulse.A, N, dt)

        # 2. phase transients via state-space IIR
        phi, _  = self._reconstruct_phase(A_bw)
        s       = self._apply_iir(phi, alpha, beta, c)
        Omega   = np.abs(A_bw)
        A_out   = Omega * np.exp(1j * s)

        # 3. noise
        if self.amplitude_noise > 0:
            A_out = A_out * (1 + self.rng.normal(0, self.amplitude_noise, N))
        if self.phase_noise_rad > 0:
            steps = self.rng.normal(0, self.phase_noise_rad * np.sqrt(dt), N)
            A_out = A_out * np.exp(1j * np.cumsum(steps))

        return Pulse(pulse.t, A_out)

    def vjp(self, pulse_in: Pulse, grad_out: np.ndarray) -> np.ndarray:
        """
        Analytical VJP through bandwidth + phase IIR.
        Noise excluded (stochastic).
        """
        N  = pulse_in.N
        dt = pulse_in.dt
        alpha, beta, c = self._iir_coeffs(dt)
        H  = self._H_bw(N, dt)

        # reproduce forward intermediates
        A_bw          = self._apply_bw(pulse_in.A, N, dt)
        phi, delta_phi = self._reconstruct_phase(A_bw)
        s             = self._apply_iir(phi, alpha, beta, c)
        Omega         = np.abs(A_bw)
        A_out         = Omega * np.exp(1j * s)

        # --- VJP: A_out = Omega * exp(i*s) ---
        e_is     = np.exp(1j * s)
        g_s      = np.real(-1j * np.conj(A_out) * grad_out)
        g_Omega  = np.real(np.conj(e_is) * grad_out)

        # --- VJP: IIR s = f(phi) ---
        g_phi = self._vjp_iir(g_s, alpha, beta, c)

        # --- VJP: amplitude channel through Omega = |A_bw| ---
        # Omega = |A_bw|, so d_Omega/d_A_bw = A_bw/|A_bw|
        safe_O              = np.where(Omega > 1e-10, Omega, 1e-10)
        g_A_bw_amplitude    = g_Omega * A_bw / safe_O

        # --- VJP: phase reconstruction phi = cumsum(delta_phi(A_bw)) ---
        g_A_bw_phase = self._vjp_reconstruct_phase(A_bw, g_phi)

        # combine amplitude and phase channels
        g_A_bw = g_A_bw_amplitude + g_A_bw_phase

        # --- VJP: bandwidth filter A_bw = IFFT(FFT(A_in) * H) ---
        grad_in = ifft(fft(g_A_bw) * H)

        return grad_in

    def parameter_summary(self):
        return {
            'bandwidth_MHz'   : self.bandwidth_MHz,
            'settling_time_us': self.settling_time_us,
            'phase_overshoot' : self.phase_overshoot,
            'amplitude_noise' : self.amplitude_noise,
            'phase_noise_rad' : self.phase_noise_rad,
        }


class FiberModel(HardwareComponent):
    """
    Single-mode UV optical fiber.

    forward():
        1. GVD  -- exp(-i*beta2/2*omega^2) in frequency domain
        2. Phase noise -- stochastic global phase shift (not in VJP)

    vjp(): analytical through GVD only.
        GVD is unitary: adjoint = conjugate transfer function.

    Parameters
    ----------
    length_mm : float
        Fiber length in mm. Paper: ~1500mm.
    dispersion_fs2_per_mm : float
        GVD coefficient in fs^2/mm. Typical UV: 100-1000 fs^2/mm.
    length_fluctuation_nm : float
        RMS fiber length fluctuation in nm.
    wavelength_nm : float
        Laser wavelength in nm. Default 302nm.
    seed : int, optional
    """

    def __init__(self, length_mm=1500.0, dispersion_fs2_per_mm=300.0,
                 length_fluctuation_nm=2.0, wavelength_nm=302.0, seed=None):
        self.length_mm             = length_mm
        self.dispersion_fs2_per_mm = dispersion_fs2_per_mm
        self.length_fluctuation_nm = length_fluctuation_nm
        self.wavelength_nm         = wavelength_nm
        self.rng = np.random.default_rng(seed)

    def _H_gvd(self, N, dt):
        """GVD transfer function H(f) = exp(-i*beta2/2*omega^2)."""
        GVD_us2 = self.dispersion_fs2_per_mm * self.length_mm * 1e-18
        freqs   = fftfreq(N, d=dt)
        omega   = 2 * np.pi * freqs
        return np.exp(-0.5j * GVD_us2 * omega**2)

    def forward(self, pulse: Pulse) -> Pulse:
        N  = pulse.N
        dt = pulse.dt
        A  = ifft(fft(pulse.A) * self._H_gvd(N, dt))
        if self.length_fluctuation_nm > 0:
            dL   = self.rng.normal(0, self.length_fluctuation_nm * 1e-9)
            dphi = 2 * np.pi * dL / (self.wavelength_nm * 1e-9)
            A    = A * np.exp(1j * dphi)
        return Pulse(pulse.t, A)

    def vjp(self, pulse_in: Pulse, grad_out: np.ndarray) -> np.ndarray:
        """
        Analytical VJP through GVD.
        GVD is unitary so adjoint uses conj(H_gvd).
        """
        N  = pulse_in.N
        dt = pulse_in.dt
        return ifft(fft(grad_out) * np.conj(self._H_gvd(N, dt)))

    def parameter_summary(self):
        return {
            'length_mm'             : self.length_mm,
            'dispersion_fs2_per_mm' : self.dispersion_fs2_per_mm,
            'length_fluctuation_nm' : self.length_fluctuation_nm,
            'wavelength_nm'         : self.wavelength_nm,
        }

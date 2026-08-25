# grape-pulse-optimizer

An implementation-focused project on GRAPE pulse optimization for neutral-atom control, with an additional differentiable hardware-correction layer for AOM and fiber distortions. I built this as a personal project while preparing for internship work at Logiqal in quantum control and numerical optimization.

The physics model and gate-optimization approach follow the six-level Rydberg Hamiltonian and GRAPE-based CZ gate design described in Ma, Liu, Peng, et al., "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit" (Thompson Lab, Princeton), [arXiv:2305.05493](https://arxiv.org/abs/2305.05493). The hardware-correction layer is likewise inspired by the fiber-coupled AOM pulse delivery and closed-loop phase correction described in that paper's beam-delivery setup.

The repository has two main pieces:

- `grape/` and `physics/`: pulse optimization for single-qubit and reduced two-qubit Rydberg models.
- `hardware/`: a differentiable model of the AOM/fiber signal chain, so pulses can be optimized against the combined distortion of the whole hardware path at once, instead of correcting each distortion source (bandwidth limits, phase transients, dispersion) independently and hoping the corrections compose cleanly.

## What the code does

- Builds six-level single-qubit and reduced-basis two-qubit Rydberg Hamiltonians.
- Optimizes amplitude and phase waveforms with GRAPE using L-BFGS-B, with optional multistart or basin-hopping variants for escaping local minima.
- Models AOM bandwidth limits, phase transients, and fiber dispersion as differentiable operators, so gradients from the fidelity objective flow back through the hardware model and directly shape the optimized waveform — the optimizer sees (and compensates for) the same distortion the hardware would actually apply.
- Includes analytical vector-Jacobian products for the main gradient paths, checked against numerical finite-difference gradients for correctness.

## Repository layout

- `run_grape.py`: CLI entry point for optimization runs.
- `grape/`: fidelity, gradient, optimizer, and regularization code.
- `physics/`: Hamiltonians and propagators.
- `hardware/`: pulse representation plus the differentiable hardware-correction components.
- `tests/`: script-style verification for gradients, propagators, and correction steps.
- `results/`: example JSON outputs from previous local runs.

## Running it

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python tests/test_all.py
python run_grape.py --lab labs/paper.yaml --gate gates/identity.yaml --no-hardware
```

## Reference

Ma, S., Liu, G., Peng, P., Zhang, B., Jandura, S., Claes, J., Burgers, A. P., Pupillo, G., Puri, S., & Thompson, J. D. "High-fidelity gates with mid-circuit erasure conversion in a metastable neutral atom qubit." [arXiv:2305.05493](https://arxiv.org/abs/2305.05493).

## Scope note

This is a serious technical project, but still exploratory. Some parts are more polished than others, and the emphasis is on readable implementations of the optimization and backpropagation ideas rather than a production-ready research codebase.

Contact: `royrliu@utexas.edu`

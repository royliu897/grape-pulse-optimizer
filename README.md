# grape-pulse-optimizer

An implementation-focused project on GRAPE pulse optimization for neutral-atom control, with an additional differentiable hardware-correction layer for AOM and fiber distortions. I built this as a personal project while preparing for internship work in quantum control and numerical optimization.

The repository has two main pieces:

- `grape/` and `physics/`: pulse optimization for single-qubit and reduced two-qubit Rydberg models.
- `hardware/`: a differentiable distortion pipeline that can be optimized through directly rather than corrected component-by-component.

## What the code does

- Builds six-level single-qubit and reduced-basis two-qubit Rydberg Hamiltonians.
- Optimizes amplitude and phase waveforms with GRAPE using L-BFGS-B and optional multistart or basin-hopping variants.
- Models AOM bandwidth limits, phase transients, and fiber dispersion as differentiable operators.
- Includes analytical vector-Jacobian products and numerical checks for the main gradient paths.

## Repository layout

- `run_grape.py`: CLI entry point for optimization runs.
- `grape/`: fidelity, gradient, optimizer, and regularization code.
- `physics/`: Hamiltonians and propagators.
- `hardware/`: pulse representation plus differentiable hardware components.
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

## Scope note

This is a serious technical project, but still exploratory. Some parts are more polished than others, and the emphasis is on readable implementations of the optimization and backpropagation ideas rather than a production-ready research codebase.

Contact: `royrliu@utexas.edu`

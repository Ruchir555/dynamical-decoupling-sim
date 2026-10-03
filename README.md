# Dynamical Decoupling Simulation

This project benchmarks basic dynamical decoupling protocols for a single qubit under dephasing noise.

The Bloch-vector model combines a constant detuning, which ideal refocusing
pulses can cancel, with Markovian transverse decay at rate `gamma_phi`, which
ideal pulses cannot reverse. The plotted x component therefore shows both the
refocused phase and the remaining exponential envelope.

## Implemented sequences

- Free evolution
- Hahn echo
- CPMG with 4 pulses
- CPMG with 8 pulses

## What it does

- Simulates qubit evolution with a Bloch-vector model
- Applies ideal pi-pulse sequences about the x-axis
- Applies pulses at their exact requested times, even when they fall between output samples
- Compares coherence preservation across protocols
- Sweeps over dephasing strength
- Generates benchmark plots and a CSV summary

## Project structure

```text
dynamical-decoupling-sim/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- src/
|   |-- bloch_sim.py
|   |-- sequences.py
|   |-- metrics.py
|   `-- run_benchmark.py
`-- results/
```

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark:

```bash
python src/run_benchmark.py
```

Run the analytical checks:

```bash
python -m unittest discover -s tests
```

## Output

Running the benchmark generates:

- `results/dd_sequences.png`
- `results/final_coherence.csv`
- `results/dd_benchmark.png`

## Future extensions

- Add finite-width pulses
- Add amplitude damping
- Add noisy pulse errors
- Add filter-function comparisons
- Add a heatmap over dephasing rate and pulse count

## Model assumptions

- The initial state is the +x Bloch state.
- The detuning and dephasing rate are constant during a run.
- Pulses are instantaneous, perfect pi rotations about x.
- `gamma_phi` is the decay rate of the transverse Bloch components.

import csv
import os
from pathlib import Path

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bloch_sim import simulate_sequence
from metrics import coherence_x, final_coherence
from sequences import cpmg_sequence, free_sequence, hahn_echo_sequence


RESULTS_DIR = PROJECT_ROOT / "results"


def plot_single_run(
    total_time: float,
    dt: float,
    omega: float,
    gamma_phi: float,
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    sequence_map = {
        "Free": free_sequence(total_time),
        "HahnEcho": hahn_echo_sequence(total_time),
        "CPMG-4": cpmg_sequence(total_time, 4),
        "CPMG-8": cpmg_sequence(total_time, 8),
    }

    plt.figure(figsize=(8, 5))

    for label, pulses in sequence_map.items():
        times, traj = simulate_sequence(
            total_time=total_time,
            dt=dt,
            omega=omega,
            gamma_phi=gamma_phi,
            pulse_times=pulses,
        )
        plt.plot(times, coherence_x(traj), label=label)

    plt.xlabel("Time")
    plt.ylabel("Coherence proxy <sigma_x>")
    plt.title("Dynamical Decoupling under Dephasing")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dd_sequences.png", dpi=300)
    plt.close()


def sweep_dephasing(
    total_time: float,
    dt: float,
    omega: float,
    gamma_values: list[float],
) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)

    labels_and_sequences = [
        ("Free", free_sequence(total_time)),
        ("HahnEcho", hahn_echo_sequence(total_time)),
        ("CPMG-4", cpmg_sequence(total_time, 4)),
        ("CPMG-8", cpmg_sequence(total_time, 8)),
    ]

    output_path = RESULTS_DIR / "final_coherence.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gamma_phi", "sequence", "final_coherence"])

        for gamma_phi in gamma_values:
            for label, pulses in labels_and_sequences:
                _, traj = simulate_sequence(
                    total_time=total_time,
                    dt=dt,
                    omega=omega,
                    gamma_phi=gamma_phi,
                    pulse_times=pulses,
                )
                writer.writerow([gamma_phi, label, final_coherence(traj)])

    return output_path


def plot_sweep(csv_path: Path) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    sequences = sorted(set(data["sequence"]))

    plt.figure(figsize=(8, 5))

    for seq in sequences:
        mask = data["sequence"] == seq
        gamma = data["gamma_phi"][mask]
        final_c = data["final_coherence"][mask]
        order = np.argsort(gamma)
        plt.plot(gamma[order], final_c[order], marker="o", label=seq)

    plt.xlabel("Dephasing rate gamma_phi")
    plt.ylabel("Final coherence <sigma_x>")
    plt.title("Benchmark of Dynamical Decoupling Sequences")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dd_benchmark.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    total_time = 10.0
    dt = 0.01
    omega = 3.0
    gammas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    plot_single_run(
        total_time=total_time,
        dt=dt,
        omega=omega,
        gamma_phi=0.12,
    )

    csv_path = sweep_dephasing(
        total_time=total_time,
        dt=dt,
        omega=omega,
        gamma_values=gammas,
    )

    plot_sweep(csv_path)
    print(f"Done. Results saved in {RESULTS_DIR}")

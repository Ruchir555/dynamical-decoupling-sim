import numpy as np


def rotation_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rotation_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def free_evolution_step(
    bloch_vec: np.ndarray,
    omega: float,
    gamma_phi: float,
    dt: float,
) -> np.ndarray:
    """
    Evolve the Bloch vector over a small timestep under z-rotation and dephasing.
    """
    rotated = rotation_z(omega * dt) @ bloch_vec
    decay = np.exp(-gamma_phi * dt)

    evolved = rotated.copy()
    evolved[0] *= decay
    evolved[1] *= decay
    return evolved


def apply_pi_x_pulse(bloch_vec: np.ndarray) -> np.ndarray:
    """
    Apply an ideal instantaneous pi pulse about the x-axis.
    """
    return rotation_x(np.pi) @ bloch_vec


def simulate_sequence(
    total_time: float,
    dt: float,
    omega: float,
    gamma_phi: float,
    pulse_times: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a qubit initialized in |+x> with Bloch vector [1, 0, 0].
    """
    n_steps = int(total_time / dt)
    times = np.linspace(0.0, total_time, n_steps + 1)

    sorted_pulse_times = sorted(pulse_times)
    pulse_index = 0
    tol = dt / 2.0

    bloch = np.array([1.0, 0.0, 0.0], dtype=float)
    trajectory = [bloch.copy()]

    for i in range(n_steps):
        current_time = times[i + 1]
        bloch = free_evolution_step(
            bloch_vec=bloch,
            omega=omega,
            gamma_phi=gamma_phi,
            dt=dt,
        )

        while (
            pulse_index < len(sorted_pulse_times)
            and abs(current_time - sorted_pulse_times[pulse_index]) <= tol
        ):
            bloch = apply_pi_x_pulse(bloch)
            pulse_index += 1

        trajectory.append(bloch.copy())

    return times, np.array(trajectory)

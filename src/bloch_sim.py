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
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if gamma_phi < 0.0:
        raise ValueError("gamma_phi must be non-negative")

    n_full_steps = int(np.floor(total_time / dt))
    times = np.arange(n_full_steps + 1, dtype=float) * dt
    if not np.isclose(times[-1], total_time):
        times = np.append(times, total_time)
    else:
        times[-1] = total_time

    sorted_pulse_times = sorted(float(t) for t in pulse_times)
    if any(t <= 0.0 or t >= total_time for t in sorted_pulse_times):
        raise ValueError("pulse times must lie strictly inside the simulation interval")
    pulse_index = 0
    bloch = np.array([1.0, 0.0, 0.0], dtype=float)
    trajectory = [bloch.copy()]
    current_time = 0.0

    for output_time in times[1:]:
        while pulse_index < len(sorted_pulse_times) and sorted_pulse_times[pulse_index] <= output_time:
            pulse_time = sorted_pulse_times[pulse_index]
            bloch = free_evolution_step(bloch, omega, gamma_phi, pulse_time - current_time)
            bloch = apply_pi_x_pulse(bloch)
            current_time = pulse_time
            pulse_index += 1

        bloch = free_evolution_step(bloch, omega, gamma_phi, output_time - current_time)
        current_time = output_time
        trajectory.append(bloch.copy())

    return times, np.array(trajectory)

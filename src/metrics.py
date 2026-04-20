import numpy as np


def coherence_x(trajectory: np.ndarray) -> np.ndarray:
    """
    Use the x-component as a proxy for coherence preservation.
    """
    return trajectory[:, 0]


def final_coherence(trajectory: np.ndarray) -> float:
    return float(trajectory[-1, 0])


def bloch_norm(trajectory: np.ndarray) -> np.ndarray:
    return np.linalg.norm(trajectory, axis=1)

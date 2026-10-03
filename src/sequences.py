def free_sequence(total_time: float) -> list[float]:
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    return []


def hahn_echo_sequence(total_time: float) -> list[float]:
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    return [total_time / 2.0]


def cpmg_sequence(total_time: float, n_pulses: int) -> list[float]:
    """
    Evenly spaced pi pulses centered in each segment.
    """
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if isinstance(n_pulses, bool) or not isinstance(n_pulses, int) or n_pulses <= 0:
        raise ValueError("n_pulses must be a positive integer")
    return [((k + 0.5) * total_time / n_pulses) for k in range(n_pulses)]

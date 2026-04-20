def free_sequence(total_time: float) -> list[float]:
    _ = total_time
    return []


def hahn_echo_sequence(total_time: float) -> list[float]:
    return [total_time / 2.0]


def cpmg_sequence(total_time: float, n_pulses: int) -> list[float]:
    """
    Evenly spaced pi pulses centered in each segment.
    """
    return [((k + 0.5) * total_time / n_pulses) for k in range(n_pulses)]

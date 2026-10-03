import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bloch_sim import simulate_sequence
from metrics import bloch_norm, coherence_x, final_coherence
from sequences import cpmg_sequence, free_sequence, hahn_echo_sequence


class BlochSimulationTests(unittest.TestCase):
    def test_free_evolution_matches_analytic_solution(self):
        total_time = 1.03
        omega = 1.7
        gamma_phi = 0.2
        times, trajectory = simulate_sequence(total_time, 0.2, omega, gamma_phi, [])

        expected_x = np.exp(-gamma_phi * times) * np.cos(omega * times)
        expected_y = np.exp(-gamma_phi * times) * np.sin(omega * times)
        np.testing.assert_allclose(trajectory[:, 0], expected_x, atol=1e-12)
        np.testing.assert_allclose(trajectory[:, 1], expected_y, atol=1e-12)
        self.assertEqual(times[-1], total_time)

    def test_echo_refocuses_constant_detuning_off_grid(self):
        total_time = 1.0
        gamma_phi = 0.13
        _, trajectory = simulate_sequence(
            total_time,
            0.17,
            2.4,
            gamma_phi,
            hahn_echo_sequence(total_time),
        )
        self.assertAlmostEqual(trajectory[-1, 0], np.exp(-gamma_phi * total_time), places=12)
        self.assertAlmostEqual(trajectory[-1, 1], 0.0, places=12)

    def test_cpmg_pulses_are_applied_at_exact_times(self):
        total_time = 1.0
        gamma_phi = 0.07
        _, trajectory = simulate_sequence(
            total_time,
            0.16,
            3.1,
            gamma_phi,
            cpmg_sequence(total_time, 8),
        )
        self.assertAlmostEqual(trajectory[-1, 0], np.exp(-gamma_phi * total_time), places=12)
        self.assertAlmostEqual(trajectory[-1, 1], 0.0, places=12)

    def test_invalid_simulation_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate_sequence(0.0, 0.1, 1.0, 0.1, [])
        with self.assertRaises(ValueError):
            simulate_sequence(1.0, 0.0, 1.0, 0.1, [])
        with self.assertRaises(ValueError):
            simulate_sequence(1.0, 0.1, 1.0, -0.1, [])
        with self.assertRaises(ValueError):
            simulate_sequence(1.0, 0.1, 1.0, 0.1, [0.0])
        with self.assertRaises(ValueError):
            simulate_sequence(1.0, 0.1, 1.0, 0.1, [1.0])

    def test_sequence_generators_validate_inputs(self):
        with self.assertRaises(ValueError):
            free_sequence(0.0)
        with self.assertRaises(ValueError):
            hahn_echo_sequence(-1.0)
        for invalid_count in (0, -1, 2.5, True):
            with self.subTest(n_pulses=invalid_count):
                with self.assertRaises(ValueError):
                    cpmg_sequence(1.0, invalid_count)

    def test_cpmg_sequence_is_symmetric(self):
        pulse_times = cpmg_sequence(2.0, 4)
        np.testing.assert_allclose(pulse_times, [0.25, 0.75, 1.25, 1.75])
        np.testing.assert_allclose(pulse_times, 2.0 - np.array(pulse_times[::-1]))

    def test_metrics_extract_expected_quantities(self):
        trajectory = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
        np.testing.assert_allclose(coherence_x(trajectory), [1.0, 0.6])
        self.assertEqual(final_coherence(trajectory), 0.6)
        np.testing.assert_allclose(bloch_norm(trajectory), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

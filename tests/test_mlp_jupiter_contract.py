import unittest
from unittest.mock import patch

import numpy as np

from src.generators import (
    DataGenerator_SolarDynamo_SDDE_Canonical,
    DataGenerator_SolarDynamo_SDDE_MLP,
)


class MlpJupiterGeneratorContractTest(unittest.TestCase):
    def test_canonical_generator_is_shared_with_fourier_cnn(self):
        self.assertIs(
            DataGenerator_SolarDynamo_SDDE_Canonical,
            DataGenerator_SolarDynamo_SDDE_MLP,
        )

    def test_original_model_uses_canonical_explicit_noise_with_continuous_delay(self):
        captured = {}

        def fake_sn(theta, eps, **kwargs):
            captured["theta"] = tuple(theta)
            captured["eps"] = np.asarray(eps).copy()
            captured["kwargs"] = kwargs
            return np.arange(4, dtype=np.float64)

        generator = DataGenerator_SolarDynamo_SDDE_Canonical(
            prng=np.random.RandomState(2718),
            model="original",
            Twarmup=2,
            Tobs=4,
            dt=0.1,
            saveat=1.0,
            tau_lims=(2.0, 2.0),
            T_lims=(3.01, 3.09),
            Nd_lims=(8.0, 8.0),
            sigma_lims=(0.02, 0.02),
            Bmax_lims=(10.0, 10.0),
        )

        with patch("sdde_model.sn_from_noise", side_effect=fake_sn):
            observation, targets, noise = next(iter(generator))

        self.assertEqual(observation.shape, (4, 1))
        self.assertEqual(targets.shape, (5,))
        self.assertEqual(len(captured["theta"]), 5)
        self.assertEqual(captured["eps"].shape, (60,))
        self.assertEqual(noise.shape, (4, 1))
        np.testing.assert_array_equal(noise[:, 0], captured["eps"][20::10][:4])
        self.assertGreater(
            abs(float(targets[1]) / 0.1 - round(float(targets[1]) / 0.1)),
            1e-5,
        )
        self.assertEqual(
            generator.simulation_backend,
            "sdde_model_sddeproblem_em_noisegrid_v2",
        )

    def test_phase_goes_to_solver_but_not_regression_targets(self):
        captured = {}

        def fake_sn(theta, eps, **kwargs):
            captured.setdefault("theta", []).append(tuple(theta))
            captured.setdefault("eps", []).append(np.asarray(eps).copy())
            captured.setdefault("kwargs", []).append(kwargs)
            return np.arange(4, dtype=np.float64)

        generator = DataGenerator_SolarDynamo_SDDE_MLP(
            prng=np.random.RandomState(31415),
            model="jupiter",
            Twarmup=2,
            Tobs=4,
            dt=0.1,
            saveat=1.0,
            tau_lims=(2.0, 2.0),
            T_lims=(3.01, 3.09),
            Nd_lims=(8.0, 8.0),
            sigma_lims=(0.02, 0.02),
            Bmax_lims=(10.0, 10.0),
            Aj_lims=(0.04, 0.06),
        )

        with patch(
            "sdde_model.solar_dynamo_jupiter.sn_from_noise",
            side_effect=fake_sn,
        ):
            iterator = iter(generator)
            observation, targets, noise = next(iterator)
            _, second_targets, second_noise = next(iterator)

        self.assertEqual(observation.shape, (4, 1))
        self.assertEqual(targets.shape, (6,))
        self.assertEqual(len(captured["theta"][0]), 7)
        np.testing.assert_allclose(captured["theta"][0][:6], targets)
        np.testing.assert_allclose(captured["theta"][1][:6], second_targets)
        self.assertGreaterEqual(captured["theta"][0][6], 0.0)
        self.assertLess(captured["theta"][0][6], 2.0 * np.pi)
        self.assertNotEqual(captured["theta"][0][6], captured["theta"][1][6])
        self.assertEqual(captured["eps"][0].shape, (60,))
        self.assertFalse(np.array_equal(captured["eps"][0], captured["eps"][1]))
        self.assertEqual(noise.shape, (4, 1))
        self.assertEqual(second_noise.shape, (4, 1))
        np.testing.assert_array_equal(noise[:, 0], captured["eps"][0][20::10][:4])

        # The canonical SDDE delay solver accepts the same continuous T prior
        # as SABC; the MLP generator must not quantize it onto the dt grid.
        self.assertGreater(abs(float(targets[1]) / 0.1 - round(float(targets[1]) / 0.1)), 1e-5)

    def test_threaded_batch_preserves_targets_noise_and_nuisance_phase(self):
        captured = {}

        def fake_batch(theta_batch, eps_batch, **kwargs):
            captured["theta"] = np.asarray(theta_batch).copy()
            captured["eps"] = np.asarray(eps_batch).copy()
            captured["kwargs"] = kwargs
            return np.tile(np.arange(4, dtype=np.float64), (2, 1))

        generator = DataGenerator_SolarDynamo_SDDE_MLP(
            prng=np.random.RandomState(31415),
            model="jupiter",
            Twarmup=2,
            Tobs=4,
            dt=0.1,
            saveat=1.0,
            tau_lims=(2.0, 2.0),
            T_lims=(3.01, 3.09),
            Nd_lims=(8.0, 8.0),
            sigma_lims=(0.02, 0.02),
            Bmax_lims=(10.0, 10.0),
            Aj_lims=(0.04, 0.06),
        )

        with patch(
            "sdde_model.solar_dynamo_jupiter.sn_from_noise_batch",
            side_effect=fake_batch,
        ):
            observations, targets, noise = generator.sample_batch(2)

        self.assertEqual(observations.shape, (2, 4, 1))
        self.assertEqual(targets.shape, (2, 6))
        self.assertEqual(noise.shape, (2, 4, 1))
        self.assertEqual(captured["theta"].shape, (2, 7))
        self.assertEqual(captured["eps"].shape, (2, 60))
        np.testing.assert_allclose(captured["theta"][:, :6], targets)
        np.testing.assert_array_equal(
            noise[:, :, 0], captured["eps"][:, 20::10][:, :4]
        )
        self.assertTrue(np.all(captured["theta"][:, 6] >= 0.0))
        self.assertTrue(np.all(captured["theta"][:, 6] < 2.0 * np.pi))
        self.assertNotEqual(captured["theta"][0, 6], captured["theta"][1, 6])

    def test_canonical_period_cannot_drift(self):
        generator = DataGenerator_SolarDynamo_SDDE_MLP(
            model="jupiter",
            Tobs=4,
            Twarmup=2,
            jupiter_period=12.0,
        )
        with self.assertRaisesRegex(ValueError, "fixes the orbital period"):
            next(iter(generator))


if __name__ == "__main__":
    unittest.main()

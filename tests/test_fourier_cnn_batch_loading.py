"""Exercise the training batch adapter without starting Julia or training."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


class FourierCnnBatchLoadingTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "train_ENCAFourierCNN_model3.py"
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        adapter = next(
            n for n in main.body
            if isinstance(n, ast.FunctionDef) and n.name == "next_batch_from_generator"
        )
        self.transform = Mock(side_effect=lambda x, n, window: x[:, :n, :])
        namespace = {
            "np": np,
            "tf": SimpleNamespace(convert_to_tensor=np.asarray),
            "args": SimpleNamespace(num_model_parameters=6, model="jupiter",
                                    num_fft_components=2, window="Hann"),
            "timeseries_to_fourier_log_amplitude": self.transform,
        }
        exec(compile(ast.Module(body=[adapter], type_ignores=[]), str(path), "exec"), namespace)
        self.load_batch = namespace["next_batch_from_generator"]
        self.raw = (
            np.arange(8, dtype=np.float32).reshape(2, 4, 1),
            np.arange(12, dtype=np.float32).reshape(2, 6),
            -np.arange(8, dtype=np.float32).reshape(2, 4, 1),
        )

    def test_batch_api_keeps_fourier_settings_targets_and_noise(self):
        source = SimpleNamespace(sample_batch=Mock(return_value=self.raw))
        x, params, noise = self.load_batch(source, 2)
        source.sample_batch.assert_called_once_with(2)
        self.assertIs(self.transform.call_args.args[0], self.raw[0])
        self.assertEqual(self.transform.call_args.args[1], 2)
        self.assertEqual(self.transform.call_args.kwargs, {"window": "Hann"})
        np.testing.assert_array_equal(x, self.raw[0][:, :2, :])
        np.testing.assert_array_equal(params, self.raw[1])
        np.testing.assert_array_equal(noise, self.raw[2])

    def test_scalar_fallback_and_batch_adapter_match(self):
        source = SimpleNamespace(sample_batch=Mock(return_value=self.raw))
        batched = self.load_batch(source, 2)
        scalar = self.load_batch(iter(zip(*self.raw)), 2)
        for expected, actual in zip(batched, scalar):
            np.testing.assert_array_equal(actual, expected)

    def test_wrong_target_width_still_rejected(self):
        source = SimpleNamespace(sample_batch=Mock(
            return_value=(self.raw[0], self.raw[1][:, :5], self.raw[2])
        ))
        with self.assertRaisesRegex(ValueError, "expected 6"):
            self.load_batch(source, 2)
        self.transform.assert_not_called()


if __name__ == "__main__":
    unittest.main()

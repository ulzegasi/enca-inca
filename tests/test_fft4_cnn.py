"""Exercise FFT4 preprocessing, training and inference without starting Julia."""

import ast
import datetime
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf
from src import enca_phase


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(filename, names, namespace, parent=None):
    path = ROOT / filename
    tree = ast.parse(path.read_text())
    if parent:
        tree = next(node for node in tree.body if getattr(node, "name", None) == parent)
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing definitions in {filename}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


SCRIPT = "train_ENCAfft4CNN_model3.py"
NS = {"tf": tf, "np": np, "VALID_WINDOWS": ("", "Hann"),
      "os": os, "datetime": datetime, "enca_phase": enca_phase}
load_definitions(SCRIPT, {
    "Architecture", "Sampler", "ExpSetup", "validate_window",
    "timeseries_to_fourier_real_imag", "timeseries_to_fourier_log_amplitude",
    "noise_to_fourier_real_imag", "validate_fft4_checkpoint",
}, NS)
Architecture = NS["Architecture"]
features = NS["timeseries_to_fourier_real_imag"]
target_transform = NS["timeseries_to_fourier_log_amplitude"]


class FFT4CnnTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.raw = self.rng.normal(size=(3, 271, 1)).astype(np.float32)
        self.noise = self.rng.normal(size=self.raw.shape).astype(np.float32)
        self.params = self.rng.uniform(0.5, 1.5, size=(3, 5)).astype(np.float32)
        self.args = SimpleNamespace(num_fft_components=100, window="Hann",
                                    num_model_parameters=5, model="original")

    def batch_loader(self):
        namespace = dict(NS, args=self.args)
        load_definitions(SCRIPT, {"next_batch_from_generator"}, namespace, parent="main")
        return namespace["next_batch_from_generator"]

    def test_observation_fft_matches_numpy_and_preserves_signed_phase(self):
        for length in (16, 17, 271):
            raw = self.raw[:, :length]
            bins = min(100, length // 2 + 1)
            for window in ("", "Hann"):
                with self.subTest(length=length, window=window):
                    windowed = raw[..., 0] * (np.hanning(length) if window else 1)
                    spectrum = np.fft.rfft(windowed, axis=1)[:, :bins]
                    expected = np.stack([spectrum.real, spectrum.imag], axis=-1) / np.sqrt(length)
                    actual = features(raw, bins, window=window).numpy()
                    np.testing.assert_allclose(actual, expected, atol=2e-6)
                    np.testing.assert_allclose(features(-raw, bins, window=window), -actual, atol=2e-6)
                    np.testing.assert_allclose(
                        target_transform(raw, bins, window=window),
                        np.log1p(np.abs(spectrum))[..., None], atol=2e-6,
                    )

    def test_batch_and_scalar_adapters_keep_input_target_and_noise_distinct(self):
        source = SimpleNamespace(sample_batch=Mock(return_value=(self.raw, self.params, self.noise)))
        load_batch = self.batch_loader()
        batched = load_batch(source, 3)
        source.sample_batch.assert_called_once_with(3)
        scalar = load_batch(iter(zip(self.raw, self.params, self.noise)), 3)
        self.assertEqual([tuple(t.shape) for t in batched],
                         [(3, 100, 2), (3, 100, 1), (3, 5), (3, 271, 1)])
        for expected, actual in zip(batched, scalar):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(batched[0], features(self.raw, 100), atol=1e-6)
        np.testing.assert_allclose(batched[1], target_transform(self.raw, 100), atol=1e-6)
        np.testing.assert_array_equal(batched[2], self.params)
        np.testing.assert_array_equal(batched[3], self.noise)
        bad_source = SimpleNamespace(sample_batch=Mock(
            return_value=(self.raw, self.params[:, :4], self.noise)))
        with self.assertRaisesRegex(ValueError, "expected 5"):
            load_batch(bad_source, 3)

    def test_encoder_has_200_inputs_and_decoder_matches_fft2(self):
        model = Architecture(5, 271, 1, 100)
        self.assertEqual(model.encoder.input_shape, (None, 100, 2))
        self.assertEqual(model.encoder.output_shape, (None, 5))
        self.assertEqual(model.decoder.output_shape, (None, 100, 1))
        latent = model.encoder(features(self.raw, 100))
        probe = tf.keras.Model(model.decoder.inputs,
                               model.decoder.get_layer("concatenate_noise_fft_and_summary").output)
        self.assertEqual(probe((latent, tf.constant(self.noise))).shape, (3, 100, 3))
        previous = {"tf": tf}
        load_definitions("train_ENCAfft2CNN_model3.py",
                         {"Architecture", "noise_to_fourier_real_imag"}, previous)
        previous_model = previous["Architecture"](5, 271, 1, 100)
        previous_model.decoder.set_weights(model.decoder.get_weights())
        np.testing.assert_allclose(
            previous_model.decoder((latent, tf.constant(self.noise))),
            model.decoder((latent, tf.constant(self.noise))), atol=1e-6,
        )

    def test_actual_training_step_uses_amplitude_target_for_both_losses(self):
        x_encoder, target, params, noise = self.batch_loader()(
            SimpleNamespace(sample_batch=Mock(return_value=(self.raw, self.params, self.noise))), 3)
        for mode in ("balanced_mse", "legacy_chisq"):
            with self.subTest(loss_mode=mode):
                self.args.loss_mode = mode
                self.args.lambda_recon = self.args.lambda_reg = 1.0
                self.args.recon_scale_eps = 1e-3
                namespace = dict(NS, args=self.args, param_widths=tf.ones(5),
                                 global_gradient_clipnorm=1e5)
                load_definitions(SCRIPT, {
                    "ChiSquareStatistic", "train_step",
                    "loss_reconstruction_fn_balanced", "loss_regress_params_fn_balanced",
                    "loss_reconstruction_fn_legacy", "loss_regress_params_fn_legacy",
                }, namespace, parent="main")
                model = Architecture(5, 271, 1, 100)
                variables = model.encoder.trainable_variables + model.decoder.trainable_variables
                before = [v.numpy().copy() for v in variables]
                prediction = model.decoder((model.encoder(x_encoder), noise)).numpy()
                if mode == "balanced_mse":
                    scale = np.maximum(np.sqrt(np.mean(target.numpy()**2, axis=1, keepdims=True)), 1e-3)
                    expected = np.mean(((target.numpy() - prediction) / scale)**2)
                else:
                    expected = np.mean((target.numpy() - prediction)**2 / np.maximum(target.numpy()**2, 1e-6))
                optimizer = tf.keras.optimizers.Adam(1e-3)
                losses, outputs, _ = namespace["train_step"](
                    model, x_encoder, target, params, noise, optimizer)
                np.testing.assert_allclose(losses[0], expected, rtol=1e-5, atol=1e-5)
                self.assertTrue(all(np.isfinite(loss.numpy()) for loss in losses))
                self.assertEqual(outputs[1].shape, target.shape)
                self.assertEqual(int(optimizer.iterations.numpy()), 1)
                self.assertTrue(any(not np.array_equal(old, new.numpy()) for old, new in zip(before, variables)))

    def test_checkpoint_roundtrip_and_all_sampler_paths(self):
        model = Architecture(5, 271, 1, 100)
        expected_latent = model.encoder(features(self.raw, 100)).numpy()
        expected_recon = model.decoder((expected_latent, self.noise)).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = tf.train.Checkpoint(encoder=model.encoder, decoder=model.decoder).save(
                str(Path(directory) / "model_ckpt"))
            restored = Architecture(5, 271, 1, 100)
            tf.train.Checkpoint(encoder=restored.encoder, decoder=restored.decoder).restore(path).assert_consumed()
            sampler = object.__new__(NS["Sampler"])
            sampler.args = SimpleNamespace(num_fft_components=100, window="Hann",
                                           ndims_latent=5, len_timeseries=271, num_noise_channels=1)
            sampler.model_obj = restored
            np.testing.assert_allclose(sampler.encode(self.raw), expected_latent, atol=1e-6)
            np.testing.assert_allclose(sampler.decode((expected_latent, self.noise)), expected_recon, atol=1e-6)
            np.testing.assert_allclose(sampler.decode((tf.constant(expected_latent), self.noise)), expected_recon, atol=1e-6)
            sampler.iterator = iter(zip(self.raw, self.params, self.noise))
            sampled_latent, sampled_noise = sampler.sample(3, return_noise_vectors=True)
            np.testing.assert_allclose(sampled_latent, expected_latent, atol=2e-6)
            np.testing.assert_array_equal(sampled_noise, self.noise)
            sampler.iterator = iter(zip(self.raw, self.params, self.noise))
            np.testing.assert_allclose(sampler.reconstruct(3), expected_recon[..., 0], atol=2e-6)

    def test_default_metadata_and_checkpoint_isolation(self):
        with patch.dict(os.environ, {"MODEL": "original"}, clear=True):
            NS["src"] = SimpleNamespace(generators=SimpleNamespace(
                DataGenerator_SolarDynamo_SDDE_Canonical=SimpleNamespace(simulation_backend="test")))
            args = NS["ExpSetup"]()
        self.assertEqual(args.ndims_latent, 5)
        self.assertEqual(args.num_fft_components, 100)
        self.assertEqual(Path(args.logdir).parent.name, "sdde_ENCAFourier4CNN_runs")
        validate = NS["validate_fft4_checkpoint"]
        validate(args)
        for key in ("representation_mode", "encoder_fft_representation",
                    "reconstruction_representation", "noise_fft_representation", "noise_window"):
            modified = vars(args).copy()
            modified.pop(key)
            with self.assertRaises(ValueError):
                validate(SimpleNamespace(**modified))
        args.representation_mode = "enca_fft2_cnn"
        with self.assertRaisesRegex(ValueError, "ENCA_FOURIER4_CNN_LOGDIR"):
            validate(args)


if __name__ == "__main__":
    unittest.main()

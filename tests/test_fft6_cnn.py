"""Exercise FFT6 preprocessing, training and inference without starting Julia."""

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


SCRIPT = "train_ENCAfft6CNN_model3.py"
NS = {"tf": tf, "np": np, "VALID_WINDOWS": ("", "Hann"),
      "os": os, "datetime": datetime, "enca_phase": enca_phase}
load_definitions(SCRIPT, {
    "Architecture", "Sampler", "ExpSetup", "validate_window",
    "timeseries_to_fourier_real_imag",
    "noise_to_fourier_real_imag", "validate_fft6_checkpoint",
}, NS)
Architecture = NS["Architecture"]
features = NS["timeseries_to_fourier_real_imag"]
noise_features = NS["noise_to_fourier_real_imag"]


class FFT6CnnTest(unittest.TestCase):
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


    def test_batch_and_scalar_adapters_keep_input_target_and_noise_distinct(self):
        source = SimpleNamespace(sample_batch=Mock(return_value=(self.raw, self.params, self.noise)))
        load_batch = self.batch_loader()
        batched = load_batch(source, 3)
        source.sample_batch.assert_called_once_with(3)
        scalar = load_batch(iter(zip(self.raw, self.params, self.noise)), 3)
        self.assertEqual([tuple(t.shape) for t in batched],
                         [(3, 100, 2), (3, 100, 2), (3, 5), (3, 271, 1)])
        for expected, actual in zip(batched, scalar):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(batched[0], features(self.raw, 100), atol=1e-6)
        np.testing.assert_allclose(batched[1], features(self.raw, 100), atol=1e-6)
        np.testing.assert_array_equal(batched[2], self.params)
        np.testing.assert_array_equal(batched[3], self.noise)
        bad_source = SimpleNamespace(sample_batch=Mock(
            return_value=(self.raw, self.params[:, :4], self.noise)))
        with self.assertRaisesRegex(ValueError, "expected 5"):
            load_batch(bad_source, 3)

    def test_decoder_receives_noise_phase_and_matches_numpy(self):
        for width in (5, 6, 7):
            model = Architecture(width, 271, 1, 100)
            self.assertEqual(model.encoder.input_shape, (None, 100, 2))
            self.assertEqual(model.encoder.output_shape, (None, width))
            self.assertEqual(model.decoder.output_shape, (None, 100, 2))
            latent = model.encoder(features(self.raw, 100))
            probe = tf.keras.Model(model.decoder.inputs,
                model.decoder.get_layer("concatenate_noise_fft_and_summary").output)
            conditioning = probe((latent, tf.constant(self.noise))).numpy()
            self.assertEqual(conditioning.shape, (3, 100, 3))
            spectrum = np.fft.rfft(self.noise[..., 0], axis=1)[:, :100] / np.sqrt(271)
            expected = np.stack([spectrum.real, spectrum.imag], axis=-1)
            np.testing.assert_allclose(conditioning[..., 1:], expected, atol=2e-6)
            shifted = np.roll(self.noise, 17, axis=1)
            shifted_features = noise_features(shifted, 100).numpy()
            np.testing.assert_allclose(np.linalg.norm(shifted_features, axis=-1),
                                       np.abs(spectrum), atol=2e-6)
            self.assertGreater(np.max(np.abs(shifted_features - expected)), 0.1)
            # Force nonzero positive weights so phase sensitivity is not a random-init accident.
            for variable in model.decoder.trainable_variables:
                variable.assign(tf.ones_like(variable) * 0.05)
            a = model.decoder((latent, tf.constant(self.noise))).numpy()
            b = model.decoder((latent, tf.constant(shifted))).numpy()
            self.assertGreater(np.max(np.abs(a - b)), 1e-5)

    def test_actual_training_step_reconstructs_complex_input(self):
        x_encoder, target, params, noise = self.batch_loader()(
            SimpleNamespace(sample_batch=Mock(return_value=(self.raw, self.params, self.noise))), 3)
        for mode, activation, width in (("complex_nmse", "leaky_relu", 5),
                                       ("complex_nmse", "leaky_relu", 6),
                                       ("complex_nmse", "leaky_relu", 7),
                                       ("mse", "leaky_relu", 5),
                                       ("balanced_mse", "relu", 5),
                                       ("legacy_chisq", "relu", 5)):
            with self.subTest(loss_mode=mode, activation=activation, width=width):
                self.args.loss_mode = mode
                self.args.lambda_recon = self.args.lambda_reg = 1.0
                self.args.recon_scale_eps = 1e-3
                namespace = dict(NS, args=self.args, param_widths=tf.ones(5),
                                 global_gradient_clipnorm=1e5)
                load_definitions(SCRIPT, {
                    "ChiSquareStatistic", "train_step", "loss_reconstruction_fn_complex_normalized",
                    "loss_reconstruction_fn_balanced", "loss_regress_params_fn_balanced",
                    "loss_reconstruction_fn_legacy", "loss_regress_params_fn_legacy",
                }, namespace, parent="main")
                model = Architecture(width, 271, 1, 100, decoder_activation=activation)
                variables = model.encoder.trainable_variables + model.decoder.trainable_variables
                before = [v.numpy().copy() for v in variables]
                prediction = model.decoder((model.encoder(x_encoder), noise)).numpy()
                if mode == "complex_nmse":
                    expected = np.mean(np.sum((target.numpy() - prediction)**2, axis=(1, 2)) /
                                       np.maximum(np.sum(target.numpy()**2, axis=(1, 2)), 200 * 1e-6))
                elif mode == "mse":
                    expected = np.mean((target.numpy() - prediction)**2)
                elif mode == "balanced_mse":
                    scale = np.maximum(np.sqrt(np.mean(target.numpy()**2, axis=1, keepdims=True)), 1e-3)
                    expected = np.mean(((target.numpy() - prediction) / scale)**2)
                else:
                    expected = 2 * np.mean((target.numpy() - prediction)**2 / np.maximum(target.numpy()**2, 1e-6))
                optimizer = tf.keras.optimizers.Adam(1e-3)
                losses, outputs, _ = namespace["train_step"](
                    model, x_encoder, target, params, noise, optimizer)
                np.testing.assert_allclose(losses[0], expected, rtol=1e-5, atol=1e-5)
                self.assertTrue(all(np.isfinite(loss.numpy()) for loss in losses))
                self.assertEqual(outputs[1].shape, target.shape)
                self.assertEqual(int(optimizer.iterations.numpy()), 1)
                self.assertTrue(any(not np.array_equal(old, new.numpy()) for old, new in zip(before, variables)))

    def test_complex_normalization_and_finite_zero_energy_gradients(self):
        self.args.recon_scale_eps = 1e-3
        namespace = dict(NS, args=self.args)
        load_definitions(SCRIPT, {"loss_reconstruction_fn_complex_normalized"}, namespace, parent="main")
        loss = namespace["loss_reconstruction_fn_complex_normalized"]
        x = features(self.raw, 100)
        pred = tf.constant(self.rng.normal(size=x.shape), dtype=tf.float32)
        np.testing.assert_allclose(loss(x, x), 0, atol=1e-7)
        np.testing.assert_allclose(loss(x, tf.zeros_like(x)), 1, atol=1e-6)
        baseline = loss(x, pred)
        # Independent sample amplitudes must not change their relative weighting.
        scales = tf.constant([0.1, 2., 100.])[:, None, None]
        np.testing.assert_allclose(loss(x * scales, pred * scales), baseline, rtol=1e-6)
        # A common complex rotation must not change the residual's norm.
        rotation = tf.constant([[0.6, -0.8], [0.8, 0.6]])
        np.testing.assert_allclose(loss(x @ rotation, pred @ rotation), baseline, rtol=1e-6)
        contributions = loss(x, pred, return_each_dim=True)
        np.testing.assert_allclose(tf.reduce_mean(list(contributions.values())), baseline, rtol=1e-6)
        variable = tf.Variable(tf.ones_like(x))
        with tf.GradientTape() as tape:
            zero_loss = loss(tf.zeros_like(x), variable)
        np.testing.assert_allclose(zero_loss, 1e6, rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(tape.gradient(zero_loss, variable))))

    def test_regression_supervises_only_first_five_latents(self):
        namespace = dict(NS, args=self.args, param_widths=tf.constant([1., 2., 3., 4., 5.]))
        load_definitions(SCRIPT, {"loss_regress_params_fn_balanced"}, namespace, parent="main")
        latent = tf.Variable(np.concatenate([self.params + 1, np.ones((3, 2)) * 99], axis=1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss = namespace["loss_regress_params_fn_balanced"](tf.constant(self.params), latent)
        np.testing.assert_allclose(loss, np.mean(1 / np.arange(1, 6)**2), rtol=1e-6)
        gradient = tape.gradient(loss, latent).numpy()
        np.testing.assert_array_equal(gradient[:, 5:], 0)
        self.assertTrue(np.all(gradient[:, :5] != 0))

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
            np.testing.assert_allclose(sampler.reconstruct(3), expected_recon, atol=2e-6)

    def test_leaky_decoder_keeps_gradient_when_last_hidden_layer_is_negative(self):
        # Reproduce the failure mechanism: every last hidden preactivation is negative.
        for activation in ("relu", "leaky_relu"):
            model = Architecture(5, 271, 1, 100, decoder_activation=activation)
            for variable in model.decoder.trainable_variables:
                variable.assign(tf.ones_like(variable) * 0.05)
            hidden = model.decoder.get_layer("dec_conv_3")
            hidden.kernel.assign(tf.zeros_like(hidden.kernel))
            hidden.bias.assign(-tf.ones_like(hidden.bias))
            latent = tf.ones((3, 5))
            with tf.GradientTape() as tape:
                prediction = model.decoder((latent, tf.constant(self.noise)))
                loss = tf.reduce_mean(tf.square(prediction - 1.0))
            gradient = tape.gradient(loss, hidden.kernel)
            self.assertTrue(np.all(np.isfinite(gradient)))
            if activation == "relu":
                self.assertEqual(float(tf.reduce_sum(tf.abs(gradient))), 0.0)
            else:
                self.assertGreater(float(tf.reduce_sum(tf.abs(gradient))), 0.0)
            self.assertEqual(model.encoder.count_params() + model.decoder.count_params(), 11727)

    def test_leaky_checkpoint_roundtrip_and_sampler_uses_saved_activation(self):
        model = Architecture(5, 271, 1, 100, decoder_activation="leaky_relu")
        latent = model.encoder(features(self.raw, 100))
        expected = model.decoder((latent, tf.constant(self.noise)))
        with tempfile.TemporaryDirectory() as directory:
            prefix = tf.train.Checkpoint(encoder=model.encoder, decoder=model.decoder).save(
                str(Path(directory) / "model_ckpt"))
            sampler = object.__new__(NS["Sampler"])
            sampler.args = SimpleNamespace(ndims_latent=5, len_timeseries=271,
                num_noise_channels=1, num_fft_components=100, decoder_activation="leaky_relu")
            with patch.dict(os.environ, {"DECODER_ACTIVATION": "relu"}):
                sampler.build_model()
            restored = sampler.model_obj
            tf.train.Checkpoint(encoder=restored.encoder, decoder=restored.decoder).restore(prefix).assert_consumed()
            np.testing.assert_allclose(restored.decoder((latent, tf.constant(self.noise))), expected, atol=1e-6)
            self.assertEqual(restored.decoder_activation, "leaky_relu")
            del sampler.args.decoder_activation
            sampler.build_model()
            self.assertEqual(sampler.model_obj.decoder_activation, "relu")

    def test_default_metadata_and_checkpoint_isolation(self):
        with patch.dict(os.environ, {"MODEL": "original"}, clear=True):
            NS["src"] = SimpleNamespace(generators=SimpleNamespace(
                DataGenerator_SolarDynamo_SDDE_Canonical=SimpleNamespace(simulation_backend="test")))
            args = NS["ExpSetup"]()
        self.assertEqual(args.loss_mode, "complex_nmse")
        self.assertEqual(args.decoder_activation, "leaky_relu")
        self.assertEqual(args.ndims_latent, 5)
        self.assertEqual(args.num_fft_components, 100)
        self.assertEqual(Path(args.logdir).parent.name, "sdde_ENCAFourier6CNN_runs")
        validate = NS["validate_fft6_checkpoint"]
        validate(args)
        legacy = SimpleNamespace(**vars(args))
        del legacy.decoder_activation
        relu = SimpleNamespace(**vars(args))
        relu.decoder_activation = "relu"
        validate(legacy, relu)
        leaky = SimpleNamespace(**vars(args))
        leaky.decoder_activation = "leaky_relu"
        validate(leaky, leaky)
        for saved, current in ((legacy, leaky), (relu, leaky), (leaky, relu)):
            with self.assertRaisesRegex(ValueError, "fresh run"):
                validate(saved, current)
        for key in ("representation_mode", "encoder_fft_representation",
                    "reconstruction_representation", "noise_fft_representation", "noise_window"):
            modified = vars(args).copy()
            modified.pop(key)
            with self.assertRaises(ValueError):
                validate(SimpleNamespace(**modified))
        previous_loss = SimpleNamespace(**vars(args))
        previous_loss.loss_mode = "mse"
        validate(previous_loss)  # Inference remains compatible.
        with self.assertRaisesRegex(ValueError, "loss_mode.*fresh run"):
            validate(previous_loss, args)
        old = SimpleNamespace(**vars(args))
        old.reconstruction_representation = "log1p_abs_rfft"
        with self.assertRaises(ValueError):
            validate(old)
        fft4 = SimpleNamespace(**vars(args))
        fft4.representation_mode = "enca_fft4_cnn"
        fft4.noise_fft_representation = "abs_rfft_ortho"
        with self.assertRaises(ValueError):
            validate(fft4)
        args.representation_mode = "enca_fft2_cnn"
        with self.assertRaisesRegex(ValueError, "ENCA_FOURIER6_CNN_LOGDIR"):
            validate(args)


if __name__ == "__main__":
    unittest.main()

"""Check the actual FFT2 training definitions without booting Julia."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import tensorflow as tf


SCRIPT = Path(__file__).resolve().parents[1] / "train_ENCAfft2CNN_model3.py"
TREE = ast.parse(SCRIPT.read_text())
NAMES = {
    "Architecture", "Sampler", "noise_to_fourier_real_imag",
    "validate_fft2_checkpoint",
}
NAMESPACE = {"tf": tf, "np": np}
exec(compile(ast.Module(
    body=[node for node in TREE.body if getattr(node, "name", None) in NAMES],
    type_ignores=[],
), str(SCRIPT), "exec"), NAMESPACE)
Architecture = NAMESPACE["Architecture"]
transform = NAMESPACE["noise_to_fourier_real_imag"]
validate_checkpoint = NAMESPACE["validate_fft2_checkpoint"]


class FFT2CnnTest(unittest.TestCase):
    def test_fft_matches_numpy_for_odd_even_lengths_and_multiple_channels(self):
        rng = np.random.default_rng(42)
        for length in (16, 17, 271):
            for channels in (1, 3):
                for bins in (4, length // 2 + 1):
                    with self.subTest(length=length, channels=channels, bins=bins):
                        noise = rng.normal(size=(2, length, channels)).astype(np.float32)
                        spectrum = np.fft.rfft(noise, axis=1, norm="ortho")[:, :bins]
                        expected = np.concatenate([spectrum.real, spectrum.imag], axis=-1)
                        actual = transform(noise, bins).numpy()
                        np.testing.assert_allclose(actual, expected, atol=1e-6)
                        if bins == length // 2 + 1:
                            restored = np.fft.irfft(
                                actual[..., :channels] + 1j * actual[..., channels:],
                                n=length, axis=1, norm="ortho",
                            )
                            np.testing.assert_allclose(restored, noise, atol=1e-6)

    def test_phase_is_preserved(self):
        impulse = np.zeros((1, 16, 1), dtype=np.float32)
        impulse[:, 1, :] = 1.0
        shifted = np.roll(impulse, 1, axis=1)
        first, second = transform(impulse, 9).numpy(), transform(shifted, 9).numpy()
        np.testing.assert_allclose(np.linalg.norm(first, axis=-1),
                                   np.linalg.norm(second, axis=-1), atol=1e-6)
        self.assertFalse(np.allclose(first, second))
        self.assertGreater(np.max(np.abs(first[..., 1])), 0.0)

    def test_decoder_convolutions_receive_real_imag_channels(self):
        model = Architecture(6, 271, 2, 100)
        noise = np.random.default_rng(7).normal(size=(2, 271, 2)).astype(np.float32)
        latent = np.zeros((2, 6), dtype=np.float32)
        probe = tf.keras.Model(
            model.decoder.inputs,
            model.decoder.get_layer("concatenate_noise_fft_and_summary").output,
        )
        conditioned = probe((latent, noise)).numpy()
        self.assertEqual(conditioned.shape, (2, 100, 5))
        np.testing.assert_allclose(conditioned[..., 1:], transform(noise, 100), atol=1e-6)
        self.assertEqual(model.decoder((latent, noise)).shape, (2, 100, 1))

    def test_training_checkpoint_roundtrip_and_sampler_decode(self):
        model = Architecture(6, 17, 1, 9)
        rng = np.random.default_rng(8)
        noise = tf.constant(rng.normal(size=(2, 17, 1)), dtype=tf.float32)
        target = tf.constant(rng.normal(size=(2, 9, 1)), dtype=tf.float32)
        variables = model.encoder.trainable_variables + model.decoder.trainable_variables
        optimizer = tf.keras.optimizers.Adam(1e-3)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                latent = model.encoder(target, training=True)
                prediction = model.decoder((latent, noise), training=True)
                loss = tf.reduce_mean(tf.square(prediction - target))
            gradients = tape.gradient(loss, variables)
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
            optimizer.apply_gradients(zip(gradients, variables))
            return loss

        self.assertTrue(np.isfinite(step().numpy()))
        latent = model.encoder(target)
        expected = model.decoder((latent, noise)).numpy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = tf.train.Checkpoint(encoder=model.encoder, decoder=model.decoder)
            path = checkpoint.save(str(Path(directory) / "model_ckpt"))
            restored = Architecture(6, 17, 1, 9)
            tf.train.Checkpoint(encoder=restored.encoder, decoder=restored.decoder).restore(
                path
            ).assert_consumed()
            np.testing.assert_allclose(restored.encoder(target), latent, atol=1e-6)
            sampler = object.__new__(NAMESPACE["Sampler"])
            sampler.model_obj = restored
            np.testing.assert_allclose(sampler.decode((latent, noise)), expected, atol=1e-6)

    def test_raw_noise_or_incomplete_checkpoint_metadata_rejected(self):
        valid = dict(representation_mode="enca_fft2_cnn",
                     noise_fft_representation="rfft_real_imag_ortho", noise_window="")
        validate_checkpoint(SimpleNamespace(**valid))
        for key in valid:
            missing = {k: v for k, v in valid.items() if k != key}
            with self.assertRaisesRegex(ValueError, "fresh ENCA_FOURIER2_CNN_LOGDIR"):
                validate_checkpoint(SimpleNamespace(**missing))
        with self.assertRaises(ValueError):
            validate_checkpoint(SimpleNamespace(**dict(valid, representation_mode="enca_fft_cnn")))

    def test_invalid_spectral_width_rejected(self):
        for bins in (0, 1, 10):
            with self.assertRaises(ValueError):
                Architecture(6, 17, 1, bins)


if __name__ == "__main__":
    unittest.main()

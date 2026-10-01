"""Opt-in phase contracts and full trainer smoke runs with a stub simulator."""

import ast
import contextlib
import datetime
import glob
import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from src import enca_phase
from src.generators import DataGenerator_SolarDynamo_SDDE_Canonical as Generator


ROOT = Path(__file__).resolve().parents[1]


def load_trainer(variant):
    path = ROOT / f"train_ENCAfft{variant}CNN_model3.py"
    tree = ast.parse(path.read_text())
    namespace = dict(tf=tf, np=np, os=os, datetime=datetime, glob=glob, json=json,
                     shutil=shutil, time=time, logging=logging, enca_phase=enca_phase,
                     __file__=str(path), VALID_WINDOWS=("", "Hann"),
                     src=SimpleNamespace(generators=SimpleNamespace(
                         DataGenerator_SolarDynamo_SDDE_Canonical=Generator)))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, tree


class PhaseGeneratorTest(unittest.TestCase):
    def generator(self, enabled):
        return Generator(model="jupiter", infer_phase=enabled, Tobs=4, Twarmup=2,
                         prng=np.random.RandomState(42))

    def test_batch_and_scalar_return_the_simulated_phase_without_changing_rng(self):
        def fake_batch(theta, eps, **kwargs):
            captured.append((np.array(theta), np.array(eps)))
            return np.tile(np.arange(4), (len(theta), 1))

        def fake_scalar(theta, eps, **kwargs):
            captured.append((np.array(theta), np.array(eps)))
            return np.arange(4)

        for batched in (True, False):
            captured = []
            name = "sn_from_noise_batch" if batched else "sn_from_noise"
            with patch(f"sdde_model.solar_dynamo_jupiter.{name}",
                       side_effect=fake_batch if batched else fake_scalar):
                with_phase = self.generator(True)
                without_phase = self.generator(False)
                if batched:
                    actual = with_phase.sample_batch(3)
                    old = without_phase.sample_batch(3)
                else:
                    actual = tuple(np.stack(v) for v in zip(*[next(iter(with_phase)) for _ in range(3)]))
                    old = tuple(np.stack(v) for v in zip(*[next(iter(without_phase)) for _ in range(3)]))
            self.assertEqual(actual[1].shape, (3, 7))
            self.assertEqual(old[1].shape, (3, 6))
            np.testing.assert_array_equal(actual[0], old[0])
            np.testing.assert_array_equal(actual[1][:, :6], old[1])
            np.testing.assert_array_equal(actual[2], old[2])
            physical = captured[0][0] if batched else np.stack([v[0] for v in captured[:3]])
            np.testing.assert_allclose(actual[1], physical, rtol=1e-6)
            self.assertTrue(np.all((actual[1][:, 6] >= 0) & (actual[1][:, 6] < 2*np.pi)))
            self.assertGreater(np.ptp(actual[1][:, 6]), 0)
        # The two APIs must also agree on every draw for a fixed seed.
        with patch("sdde_model.solar_dynamo_jupiter.sn_from_noise_batch", side_effect=fake_batch), \
             patch("sdde_model.solar_dynamo_jupiter.sn_from_noise", side_effect=fake_scalar):
            gen = self.generator(True)
            batch = gen.sample_batch(3)
            gen = self.generator(True)
            scalar = tuple(np.stack(v) for v in zip(*[next(iter(gen)) for _ in range(3)]))
        for a, b in zip(batch, scalar):
            np.testing.assert_array_equal(a, b)

    def test_phase_option_is_explicit_and_only_valid_for_jupiter(self):
        with self.assertRaisesRegex(ValueError, "jupiter"):
            Generator(model="original", infer_phase=True)
        with self.assertRaisesRegex(ValueError, "boolean"):
            Generator(model="jupiter", infer_phase="false")
        with patch.dict(os.environ, {"INFER_PHASE": "true"}):
            self.assertFalse(Generator(model="jupiter").infer_phase)


class PhaseTrainingTest(unittest.TestCase):
    def test_configuration_and_checkpoint_compatibility(self):
        for variant in (2, 4):
            ns, _ = load_trainer(variant)
            for model, enabled, physical, supervised in (
                ("original", False, 5, 5), ("jupiter", False, 6, 6), ("jupiter", True, 7, 8),
            ):
                with patch.dict(os.environ, {"MODEL": model, "INFER_PHASE": str(enabled)}, clear=True):
                    args = ns["ExpSetup"]()
                self.assertEqual(args.num_model_parameters, physical)
                self.assertEqual(args.num_supervised_parameters, supervised)
                self.assertEqual(args.ndims_latent, supervised)
                enca_phase.validate_phase_checkpoint(args, args)
                if enabled:
                    self.assertEqual(args.phase_prior, [0.0, 2*np.pi])
                    self.assertEqual(args.supervised_names[-2:], ["sin_phi", "cos_phi"])
                    for key in enca_phase.phase_configuration(model, enabled):
                        missing = vars(args).copy()
                        missing.pop(key)
                        with self.assertRaises(ValueError):
                            enca_phase.validate_phase_checkpoint(SimpleNamespace(**missing), args)
                else:
                    # Pre-option checkpoints omitted these fields and remain resumable.
                    legacy = SimpleNamespace(model=model, num_model_parameters=physical,
                                             ndims_latent=supervised)
                    enca_phase.validate_phase_checkpoint(legacy, args)
                    if model == "jupiter":
                        requested = SimpleNamespace(infer_phase=True)
                        with self.assertRaisesRegex(ValueError, "fresh run"):
                            enca_phase.validate_phase_checkpoint(legacy, requested)
            for env in ({"MODEL": "original", "INFER_PHASE": "true"},
                        {"MODEL": "jupiter", "INFER_PHASE": "true", "NDIMS_LATENT": "7"},
                        {"MODEL": "jupiter", "INFER_PHASE": "typo"}):
                with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                    ns["ExpSetup"]()

    def test_phase_loss_wraps_and_counts_the_pair_once(self):
        theta = tf.constant([[1, 2, 3, 4, 5, 6, 0]], dtype=tf.float32)
        targets = enca_phase.supervised_targets(theta, True)
        np.testing.assert_array_equal(targets, [[1, 2, 3, 4, 5, 6, 0, 1]])
        wrong_phase = tf.constant([[1, 2, 3, 4, 5, 6, 1, 0]], dtype=tf.float32)
        self.assertAlmostEqual(float(enca_phase.phase_pair_mse(theta, wrong_phase)), 0.25)
        for variant in (2, 4):
            ns, tree = load_trainer(variant)
            main = next(n for n in tree.body if getattr(n, "name", None) == "main")
            names = {"loss_regress_params_fn_balanced", "loss_regress_params_fn_legacy", "ChiSquareStatistic"}
            defs = [n for n in main.body if getattr(n, "name", None) in names]
            ns.update(args=SimpleNamespace(infer_phase=True), param_widths=tf.ones(6))
            exec(compile(ast.Module(body=defs, type_ignores=[]), "losses", "exec"), ns)
            for loss_name in ("loss_regress_params_fn_balanced", "loss_regress_params_fn_legacy"):
                loss = ns[loss_name]
                self.assertAlmostEqual(float(loss(theta, targets)), 0.0)
                self.assertAlmostEqual(float(loss(theta, wrong_phase)), 0.25/7, places=6)
                prediction = tf.Variable(tf.concat([wrong_phase, [[99.0]]], axis=-1))
                with tf.GradientTape() as tape:
                    value = loss(theta, prediction)
                grad = tape.gradient(value, prediction).numpy()
                self.assertTrue(np.isfinite(grad).all())
                self.assertTrue(np.all(grad[0, 6:8] != 0))
                self.assertEqual(grad[0, 8], 0)
        theta_near_wrap = tf.constant([[1, 2, 3, 4, 5, 6, 2*np.pi-0.01]], dtype=tf.float32)
        latent = tf.constant([[1, 2, 3, 4, 5, 6, np.sin(0.01), np.cos(0.01)]], dtype=tf.float32)
        np.testing.assert_allclose(enca_phase.wrapped_phase_error(theta_near_wrap, latent), [0.02], atol=1e-6)
        self.assertLess(float(enca_phase.phase_pair_mse(theta_near_wrap, latent)), 1e-4)

    def test_both_full_training_loops_save_resume_and_load_phase_checkpoints(self):
        class FakeGenerator:
            simulation_backend = "test_phase_backend"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def sample_batch(self, batch_size):
                rng = np.random.RandomState(51)
                size = int(self.kwargs["Tobs"] / self.kwargs["saveat"])
                noise = rng.normal(size=(batch_size, size, 1)).astype(np.float32)
                theta = np.tile([2., 3., 8., .02, 10., .05, 1.2], (batch_size, 1)).astype(np.float32)
                width = 7 if self.kwargs.get("infer_phase") else (6 if self.kwargs["model"] == "jupiter" else 5)
                return np.abs(noise), theta[:, :width], noise

            def __iter__(self):
                while True:
                    yield tuple(value[0] for value in self.sample_batch(1))

        for variant in (2, 4):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                ns, _ = load_trainer(variant)
                ns["src"].generators.DataGenerator_SolarDynamo_SDDE_Canonical = FakeGenerator
                with patch.dict(os.environ, {"MODEL": "jupiter", "INFER_PHASE": "true"}, clear=True):
                    args = ns["ExpSetup"]()
                args.logdir = directory
                args.Tobs = args.len_timeseries = 17
                args.num_fft_components = 9
                args.batch_size = 3
                args.max_training_steps = 2
                args.freq_log = 1
                ns["ExpSetup"] = lambda: args
                with contextlib.redirect_stdout(io.StringIO()):
                    ns["main"]()
                saved = json.loads((Path(directory) / "hyper_parameters.json").read_text())
                self.assertEqual(saved["num_model_parameters"], 7)
                self.assertEqual(saved["num_supervised_parameters"], 8)
                self.assertEqual(saved["phase_reference"], "simulator_t0_before_warmup")
                # Full resume exercises optimizer restore, logging, and metadata checks.
                args.max_training_steps = 4
                with contextlib.redirect_stdout(io.StringIO()):
                    ns["main"]()
                self.assertTrue((Path(directory) / "model_ckpt-4.index").is_file())
                # Explicit checkpoint loading must ignore conflicting shell defaults.
                with patch.dict(os.environ, {"MODEL": "original", "INFER_PHASE": "true"}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    sampler = ns["Sampler"](logdir=directory)
                self.assertTrue(sampler.args.infer_phase)
                self.assertEqual(sampler.model_obj.encoder.output_shape, (None, 8))
                generator, _ = sampler.build_custom_generator(return_generator=True)
                self.assertTrue(generator.kwargs["infer_phase"])
                raw = np.ones((2, 17, 1), dtype=np.float32)
                latent = sampler.encode(raw)
                self.assertEqual(sampler.decode((latent, raw)).shape, (2, 9, 1))
                # Reject mode switches before rewriting checkpoint metadata.
                before = (Path(directory) / "hyper_parameters.json").read_bytes()
                args.__dict__.update(enca_phase.phase_configuration("jupiter", False))
                args.ndims_latent = 6
                with self.assertRaisesRegex(ValueError, "fresh run"), contextlib.redirect_stdout(io.StringIO()):
                    ns["main"]()
                self.assertEqual((Path(directory) / "hyper_parameters.json").read_bytes(), before)

    def test_launcher_modes_and_phase_run_names(self):
        # Execute only the pure configuration block, never modules/srun/Slurm.
        for variant in (2, 4):
            script = (ROOT / f"runtraining_gpu_encafourier{variant}cnn.sh").read_text()
            config = script[script.index('# Training settings'):script.index('export JULIA_DEPOT_PATH')]
            export_line = next(line for line in script.splitlines() if line.startswith(f'export ENCA_FOURIER{variant}_CNN_LOGDIR='))
            probe = config + '\nRUNSTAMP=test\n' + export_line + f'\necho "$NDIMS_LATENT $ENCA_FOURIER{variant}_CNN_LOGDIR"\n'
            for model, flag, width, tag in (("original", "false", 5, "_z5"),
                                           ("jupiter", "false", 6, "_jupiter_z6"),
                                           ("jupiter", "true", 8, "_jupiter_phase_z8")):
                # Exercise settings as edited in the script, not shell overrides.
                edited = probe.replace('export MODEL="jupiter"', f'export MODEL="{model}"', 1)
                edited = edited.replace('export INFER_PHASE="true"', f'export INFER_PHASE="{flag}"', 1)
                env = dict(PATH=os.environ["PATH"], MODEL="original", INFER_PHASE="false", NDIMS_LATENT="5")
                result = subprocess.run(["bash", "-c", edited], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.startswith(f"{width} "))
                self.assertIn(tag, result.stdout)
            invalid = subprocess.run(["bash", "-c", probe.replace('export MODEL="jupiter"', 'export MODEL="original"', 1)], capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()

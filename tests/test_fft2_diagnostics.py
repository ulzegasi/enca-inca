"""Exercise diagnostic CLIs with real TensorFlow checkpoints and a mock simulator."""
import ast
import datetime
import glob
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, namespace):
    tree = ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(
        n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]), str(path), "exec"), namespace)


TRAINER = dict(tf=tf, np=np, VALID_WINDOWS=("", "Hann"))
definitions(ROOT / "train_ENCAfft2CNN_model3.py", TRAINER)


class Generator:
    simulation_backend = "sdde_model_sddeproblem_em_noisegrid_v2"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __iter__(self):
        k = self.kwargs
        rng = k["prng"]
        while True:
            names = ["tau", "T", "Nd", "sigma", "Bmax"]
            if k["model"] == "jupiter":
                names.append("Aj")
            theta = np.array([rng.uniform(*k[n + "_lims"]) for n in names], dtype=np.float32)
            noise = rng.normal(size=(271, 1)).astype(np.float32)
            yield np.abs(noise) * 10, theta, noise


class FFT2DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.hp = dict(model="original", representation_mode="enca_fft2_cnn",
            noise_fft_representation="rfft_real_imag_ortho", noise_window="", window="Hann",
            simulation_backend=Generator.simulation_backend, num_model_parameters=5,
            ndims_latent=5, Tobs=271, Twarmup=200, dt=0.1, saveat=1,
            num_fft_components=100, tau_lims=[.1,10], T_lims=[.1,10],
            Nd_lims=[1,15], sigma_lims=[.005,.05], Bmax_lims=[1,15])
        (self.run / "hyper_parameters.json").write_text(json.dumps(self.hp))
        model = TRAINER["Architecture"](5,271,1,100)
        for v in model.encoder.trainable_variables:
            v.assign(tf.zeros_like(v))
        ckpt = tf.train.Checkpoint(encoder=model.encoder, decoder=model.decoder)
        _ = ckpt.save_counter  # Training CheckpointManager includes this field.
        model.encoder.get_layer("latent_channels").bias.assign(tf.ones(5)*3)
        ckpt.write(str(self.run / "model_best_ckpt-7"))
        model.encoder.get_layer("latent_channels").bias.assign(tf.ones(5)*4)
        ckpt.write(str(self.run / "model_ckpt-9"))

    def namespace(self, prefix):
        ns = dict(TRAINER, argparse=__import__('argparse'), os=os, json=json, glob=glob,
            time=time, datetime=datetime, plt=plt, SimpleNamespace=SimpleNamespace,
            BASE_PARAM_NAMES=["tau","T","Nd","sigma","Bmax"],
            src=SimpleNamespace(generators=SimpleNamespace(
                DataGenerator_SolarDynamo_SDDE_Canonical=Generator)))
        definitions(ROOT / f"{prefix}_test_encafourier2cnn.py", ns)
        return ns

    def test_diagonal_restores_best_and_last_and_saves_metrics(self):
        ns = self.namespace("diag")
        for option, expected in (("--best",3), ("--last",4)):
            out = self.run / option[2:]
            with patch.object(sys, "argv", ["diag", "--logdir", str(self.run),
                    "--nsamples", "8", "--batch", "3", "--outdir", str(out), option]):
                ns["main"]()
            data = np.load(next(out.glob('*.npz')))
            np.testing.assert_allclose(data['predicted_params'], expected)
            self.assertEqual(data['true_params'].shape, (8,5))
            report = json.loads(next(out.glob('*.json')).read_text())
            self.assertEqual(set(report['metrics']), set(ns['BASE_PARAM_NAMES']))
            self.assertTrue(next(out.glob('*.png')).stat().st_size > 1000)

    def test_reconstruction_and_saved_prior_validation(self):
        ns = self.namespace("recon")
        argv = ["recon", "--logdir", str(self.run), "--tau", "2", "--T", "3",
                "--Nd", "8", "--sigma", ".02", "--Bmax", "10", "--nseeds", "3"]
        with patch.object(sys, "argv", argv):
            ns["main"]()
        data = np.load(next((self.run/'diagnostics').glob('*.npz')))
        self.assertEqual(data['target'].shape,(3,100))
        self.assertTrue(np.isfinite(data['reconstruction']).all())
        self.assertFalse(np.array_equal(data['target'][0],data['target'][1]))
        argv[argv.index('.02')] = '.2'
        with patch.object(sys,"argv",argv), self.assertRaisesRegex(ValueError,'outside saved prior'):
            ns['main']()

    def test_wrong_architecture_and_phase_mode_rejected(self):
        for prefix in ('diag','recon'):
            ns=self.namespace(prefix)
            for changes in ({'representation_mode':'enca_fft_cnn'}, {'infer_phase':True}):
                hp=dict(self.hp,**changes)
                (self.run/'hyper_parameters.json').write_text(json.dumps(hp))
                with self.assertRaises(ValueError):
                    ns['load_hparams'](str(self.run))


if __name__ == '__main__':
    unittest.main()

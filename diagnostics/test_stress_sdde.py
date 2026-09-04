"""Fast driver tests; do not initialize Julia, TensorFlow, or Slurm."""
import unittest
from unittest.mock import Mock

import numpy as np

from diagnostics import stress_sdde as stress


class StressDriverTests(unittest.TestCase):
    def batch(self, batch_size=2):
        return (np.ones((batch_size, 271, 1), dtype=np.float32),
                np.ones((batch_size, 6), dtype=np.float32),
                np.zeros((batch_size, 271, 1), dtype=np.float32))

    def test_defaults_match_crashed_training(self):
        args = stress.parse_args(["--threads", "16"])
        self.assertEqual((args.seed, args.batch_size, args.seconds), (1822, 300, 3600))
        self.assertFalse(args.with_tensorflow)

    def test_invalid_limit_rejected(self):
        with self.assertRaises(SystemExit):
            stress.parse_args(["--threads", "0"])

    def test_validation_checks_shapes_and_finiteness(self):
        stress.validate_batch(self.batch(), 2, np)
        bad = self.batch()
        bad[0][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            stress.validate_batch(bad, 2, np)
        with self.assertRaises(ValueError):
            stress.validate_batch(self.batch(), 3, np)

    def test_batch_cap_and_prefix_hash_repeatability(self):
        args = stress.parse_args(["--threads", "1", "--batches", "2", "--batch-size", "2"])
        hashes = []
        for _ in range(2):
            gen = Mock()
            gen.sample_batch.return_value = self.batch()
            records = []
            result = stress.run_batches(args, gen, np, clock=lambda: 0,
                                        log=lambda event, **fields: records.append((event, fields)))
            self.assertEqual(result, 2)
            self.assertEqual(gen.sample_batch.call_count, 2)
            self.assertEqual(records[-1][1]["reason"], "batch_limit")
            hashes.append(records[-1][1]["prefix_sha256"])
        self.assertEqual(*hashes)

    def test_expired_time_cap_does_not_start_batch(self):
        args = stress.parse_args(["--threads", "1", "--seconds", "1"])
        gen = Mock()
        clock = iter([0, 2, 2])
        log = Mock()
        self.assertEqual(stress.run_batches(args, gen, np, clock=lambda: next(clock), log=log), 0)
        gen.sample_batch.assert_not_called()
        self.assertEqual(log.call_args.kwargs["reason"], "time_limit")

    def test_simulator_error_is_not_silently_retried(self):
        args = stress.parse_args(["--threads", "1"])
        gen = Mock()
        gen.sample_batch.side_effect = RuntimeError("test failure")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            stress.run_batches(args, gen, np, clock=lambda: 0, log=Mock())
        self.assertEqual(gen.sample_batch.call_count, 1)


if __name__ == "__main__":
    unittest.main()

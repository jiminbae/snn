import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_temporal_reliability_confirmatory_nmnist as launcher


class GpuParsingTests(unittest.TestCase):
    def test_parse_four_gpus(self):
        self.assertEqual(launcher.parse_gpu_list("0,1,2,3"), ["0", "1", "2", "3"])

    def test_duplicate_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            launcher.parse_gpu_list("0,0,1")

    def test_invalid_gpu_strings_are_rejected(self):
        for value in ("", "0,,2", "a,b", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    launcher.parse_gpu_list(value)

    def test_smoke_and_parallel_are_incompatible(self):
        with self.assertRaises(SystemExit):
            launcher.parse_args(["out", "--smoke", "--parallel-gpus", "0,1"])


class JobTests(unittest.TestCase):
    def test_builds_exact_nine_jobs(self):
        jobs = launcher.build_jobs(Path("out"))
        self.assertEqual(len(jobs), 9)
        self.assertEqual(
            [(job.label, job.seed) for job in jobs],
            [
                ("final_ce", 3), ("final_ce", 4), ("final_ce", 5),
                ("symmetric_kl", 3), ("symmetric_kl", 4), ("symmetric_kl", 5),
                ("selective_regression_thr0.6", 3),
                ("selective_regression_thr0.6", 4),
                ("selective_regression_thr0.6", 5),
            ],
        )

    def test_threshold_is_only_on_selective_jobs(self):
        for job in launcher.build_jobs(Path("out")):
            command = job.command("python")
            has_threshold = "--temporal-confidence-threshold" in command
            self.assertEqual(has_threshold, job.method == "selective_regression")
            if has_threshold:
                index = command.index("--temporal-confidence-threshold")
                self.assertEqual(command[index + 1], "0.6")

    def test_completed_runs_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = launcher.build_jobs(Path(directory))
            completed = jobs[0]
            completed.run_dir.mkdir(parents=True)
            for filename in launcher.COMPLETION_FILES:
                (completed.run_dir / filename).touch()
            pending, done = launcher.pending_jobs(jobs)
            self.assertEqual(done, [completed])
            self.assertNotIn(completed, pending)


class SchedulerTests(unittest.TestCase):
    def test_max_concurrency_does_not_exceed_gpu_count(self):
        jobs = launcher.build_jobs(Path("out"))
        lock = threading.Lock()
        active = 0
        maximum = 0

        def runner(job, gpu):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return True

        outcomes = launcher.run_job_pool(jobs, ["0", "1", "2", "3"], runner)
        self.assertEqual(len(outcomes), 9)
        self.assertLessEqual(maximum, 4)

    def test_one_gpu_preserves_serial_job_order(self):
        jobs = launcher.build_jobs(Path("out"))
        order = []

        def runner(job, gpu):
            order.append(job)
            return True

        launcher.run_job_pool(jobs, ["0"], runner)
        self.assertEqual(order, jobs)

    def test_each_gpu_runs_at_most_one_job(self):
        jobs = launcher.build_jobs(Path("out"))
        lock = threading.Lock()
        active = {gpu: 0 for gpu in ("0", "1", "2", "3")}

        def runner(job, gpu):
            with lock:
                active[gpu] += 1
                self.assertEqual(active[gpu], 1)
            time.sleep(0.005)
            with lock:
                active[gpu] -= 1
            return True

        launcher.run_job_pool(jobs, list(active), runner)


class LauncherFlowTests(unittest.TestCase):
    def run_main(self, outcome_success):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        jobs = launcher.build_jobs(Path(temporary.name))
        outcomes = [(job, "0", outcome_success) for job in jobs]
        aggregate_result = Mock(returncode=0)
        patches = (
            patch.object(launcher, "validate_gpu_indices"),
            patch.object(launcher, "prepare_nmnist_data"),
            patch.object(launcher, "run_job_pool", return_value=outcomes),
            patch.object(launcher.subprocess, "run", return_value=aggregate_result),
            patch.dict(os.environ, {"PYTHON_BIN": os.sys.executable}, clear=False),
        )
        entered = []
        for item in patches:
            entered.append(item.__enter__())
            self.addCleanup(item.__exit__, None, None, None)
        result = launcher.main(
            [temporary.name, "--parallel-gpus", "0,1,2,3"]
        )
        return result, entered[3]

    def test_failed_job_prevents_aggregate(self):
        result, run_mock = self.run_main(False)
        self.assertEqual(result, 1)
        run_mock.assert_not_called()

    def test_success_runs_aggregate_once(self):
        result, run_mock = self.run_main(True)
        self.assertEqual(result, 0)
        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertIn("aggregate_temporal_reliability_confirmatory.py", command)


if __name__ == "__main__":
    unittest.main()


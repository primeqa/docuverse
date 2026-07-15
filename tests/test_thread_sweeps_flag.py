import argparse
import unittest

from scripts.run_retrieval_experiment import DEFAULT_SETTINGS, _cli_overrides


def _ns(**overrides):
    """Build an argparse.Namespace with every attribute _cli_overrides reads,
    defaulted to the 'not passed' value, then apply overrides.

    NOTE: keep the `base` dict below in sync with the attributes
    `_cli_overrides` accesses — a missing key would surface as an
    AttributeError only in the real CLI path, not here.
    """
    base = dict(
        dataset_name=None, queries_jsonl=None, corpus_file=None,
        top_k=None, alpha=None, speed_queries=None, warmup=None,
        workers=None, milvus_uri=None, index_root=None, seed=None,
        results_db=None, threads=None, fagin_epsilon=None,
        fagin_schedule=None, no_milvus=False, no_faiss=False,
        no_fagin=False, no_simdq=False, skip_quality=False,
        skip_speed=False, out=None, thread_sweeps=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestThreadSweepsFlag(unittest.TestCase):
    def test_default_is_off(self):
        self.assertIs(DEFAULT_SETTINGS["run_thread_sweeps"], False)

    def test_flag_sets_override(self):
        ov = _cli_overrides(_ns(thread_sweeps=True))
        self.assertIs(ov["settings.run_thread_sweeps"], True)

    def test_absent_flag_no_override(self):
        ov = _cli_overrides(_ns(thread_sweeps=False))
        self.assertNotIn("settings.run_thread_sweeps", ov)

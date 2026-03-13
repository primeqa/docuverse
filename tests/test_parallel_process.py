"""Tests for parallel_process and helpers in docuverse.utils."""
import pytest
from docuverse.utils import parallel_process, _report_failed_items


# ---------------------------------------------------------------------------
# Helpers used across tests
# ---------------------------------------------------------------------------

def _identity(x):
    """Return input wrapped in a list (simulates typical process_func)."""
    return [x]


def _double(x):
    """Return list with doubled value."""
    return [x * 2]


def _upper(x):
    """Uppercase a string, return as single-element list of dicts."""
    return [{"text": x.upper()}]


def _length_post(item):
    """Post-function: compute length of 'text' field."""
    return len(item["text"])


def _explode_on_three(x):
    """Raises on input == 3, works otherwise."""
    if x == 3:
        raise ValueError(f"Cannot process value {x}")
    return [x]


# ---------------------------------------------------------------------------
# Single-threaded path (num_threads <= 1)
# ---------------------------------------------------------------------------

class TestSingleThreaded:
    def test_basic_processing(self):
        data = [1, 2, 3, 4, 5]
        results = parallel_process(_double, data, num_threads=1, msg="test")
        assert results == [[2], [4], [6], [8], [10]]

    def test_empty_data(self):
        results = parallel_process(_identity, [], num_threads=1, msg="test")
        assert results == []

    def test_single_item(self):
        results = parallel_process(_identity, ["hello"], num_threads=1, msg="test")
        assert results == [["hello"]]

    def test_preserves_order(self):
        data = list(range(50))
        results = parallel_process(_identity, data, num_threads=1, msg="test")
        assert results == [[i] for i in range(50)]

    def test_post_func_dict(self):
        data = ["hello", "world"]
        results = parallel_process(_upper, data, num_threads=1,
                                   post_func=_length_post, post_label="len",
                                   msg="test")
        assert results[0] == [{"text": "HELLO", "len": 5}]
        assert results[1] == [{"text": "WORLD", "len": 5}]

    def test_zero_threads_uses_single(self):
        """num_threads=0 should also use the single-threaded path."""
        data = [10, 20]
        results = parallel_process(_double, data, num_threads=0, msg="test")
        assert results == [[20], [40]]


# ---------------------------------------------------------------------------
# Multi-threaded path (num_threads > 1)
# ---------------------------------------------------------------------------

class TestMultiThreaded:
    def test_basic_processing(self):
        data = [1, 2, 3, 4, 5]
        results = parallel_process(_double, data, num_threads=2, msg="test")
        assert results == [[2], [4], [6], [8], [10]]

    def test_preserves_order(self):
        """Results must come back in original order despite parallel execution."""
        data = list(range(30))
        results = parallel_process(_identity, data, num_threads=4, msg="test")
        assert results == [[i] for i in range(30)]

    def test_many_items_few_threads(self):
        data = list(range(100))
        results = parallel_process(_identity, data, num_threads=2, msg="test")
        assert len(results) == 100
        assert all(r == [i] for i, r in enumerate(results))

    def test_post_func_dict(self):
        data = ["foo", "barbaz"]
        results = parallel_process(_upper, data, num_threads=2,
                                   post_func=_length_post, post_label="len",
                                   msg="test")
        assert results[0] == [{"text": "FOO", "len": 3}]
        assert results[1] == [{"text": "BARBAZ", "len": 6}]

    def test_single_item_multithread(self):
        """Even with 1 item and multiple threads, should work correctly."""
        results = parallel_process(_identity, ["only"], num_threads=3, msg="test")
        assert results == [["only"]]

    def test_string_data(self):
        data = ["alpha", "beta", "gamma"]
        results = parallel_process(_upper, data, num_threads=2, msg="test")
        assert results[0] == [{"text": "ALPHA"}]
        assert results[1] == [{"text": "BETA"}]
        assert results[2] == [{"text": "GAMMA"}]

    def test_worker_exception_returns_empty_list(self):
        """Items that raise exceptions should get [] and not crash the run."""
        data = [1, 2, 3, 4, 5]
        results = parallel_process(_explode_on_three, data, num_threads=2, msg="test")
        assert results[0] == [1]
        assert results[1] == [2]
        assert results[2] == []  # failed item
        assert results[3] == [4]
        assert results[4] == [5]


# ---------------------------------------------------------------------------
# _report_failed_items
# ---------------------------------------------------------------------------

class TestReportFailedItems:
    def test_no_failures(self, capsys):
        results = [["a"], ["b"], ["c"]]
        data = ["a", "b", "c"]
        _report_failed_items(results, data, num_docs=3, collected=3)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_crashes_reported(self, capsys):
        """Items with None results (worker crashed) are reported."""
        results = [["ok"], None, ["ok"]]
        data = [{"id": "doc0"}, {"id": "doc1"}, {"id": "doc2"}]
        _report_failed_items(results, data, num_docs=3, collected=2)
        captured = capsys.readouterr()
        assert "1 crashed" in captured.out
        assert "doc1" in captured.out

    def test_exceptions_reported_with_traceback(self, capsys):
        """Items with errors dict entries show the traceback."""
        results = [["ok"], [], ["ok"]]
        data = [{"id": "d0"}, {"id": "d1"}, {"id": "d2"}]
        errors = {1: "Traceback (most recent call last):\n  ...\nValueError: bad value\n"}
        _report_failed_items(results, data, num_docs=3, collected=3, errors=errors)
        captured = capsys.readouterr()
        assert "1 raised exceptions" in captured.out
        assert "ValueError: bad value" in captured.out

    def test_exceptions_grouped_by_traceback(self, capsys):
        """Same exception from multiple items is grouped."""
        tb = "Traceback (most recent call last):\n  ...\nKeyError: 'x'\n"
        results = [[], ["ok"], []]
        data = ["a", "b", "c"]
        errors = {0: tb, 2: tb}
        _report_failed_items(results, data, num_docs=3, collected=3, errors=errors)
        captured = capsys.readouterr()
        assert "Distinct exceptions (1)" in captured.out
        assert "Affected 2 item(s)" in captured.out

    def test_mixed_crashes_and_exceptions(self, capsys):
        """Both crashes (None) and exceptions are reported."""
        results = [None, [], ["ok"]]
        data = ["a", "b", "c"]
        errors = {1: "Traceback ...\nRuntimeError: boom\n"}
        _report_failed_items(results, data, num_docs=3, collected=2, errors=errors)
        captured = capsys.readouterr()
        assert "1 crashed" in captured.out
        assert "1 raised exceptions" in captured.out
        assert "RuntimeError: boom" in captured.out

    def test_crash_dict_no_id(self, capsys):
        results = [None, ["ok"]]
        data = [{"text": "some long text here"}, {"text": "other"}]
        _report_failed_items(results, data, num_docs=2, collected=1)
        captured = capsys.readouterr()
        assert "1 crashed" in captured.out
        assert "some long text here" in captured.out

    def test_crash_string_data(self, capsys):
        results = [None, None, ["ok"]]
        data = ["first", "second", "third"]
        _report_failed_items(results, data, num_docs=3, collected=1)
        captured = capsys.readouterr()
        assert "2 crashed" in captured.out
        assert "first" in captured.out
        assert "second" in captured.out

    def test_max_display_limit(self, capsys):
        """When more than 20 crashes, only first 20 should be shown."""
        n = 25
        results = [None] * n
        data = [{"id": f"doc{i}"} for i in range(n)]
        _report_failed_items(results, data, num_docs=n, collected=0)
        captured = capsys.readouterr()
        assert "and 5 more" in captured.out
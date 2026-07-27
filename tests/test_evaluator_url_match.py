import unittest

from docuverse.engines.data_template import default_query_template
from docuverse.engines.search_engine_config_params import EvaluationArguments
from docuverse.engines.search_queries import SearchQueries
from docuverse.engines.search_result import SearchResult
from docuverse.utils import normalize_doc_url
from docuverse.utils.evaluator import EvaluationEngine


class TestNormalizeDocUrl(unittest.TestCase):
    def test_dotted_version_replaced(self):
        self.assertEqual(
            normalize_doc_url("https://www.ibm.com/docs/en/zos/2.5.0?topic=foo"),
            "https://www.ibm.com/docs/en/zos/latest?topic=foo",
        )

    def test_single_integer_version_replaced(self):
        self.assertEqual(
            normalize_doc_url("https://www.ibm.com/docs/en/db2-for-zos/13?topic=bar"),
            "https://www.ibm.com/docs/en/db2-for-zos/latest?topic=bar",
        )

    def test_multi_segment_product_replaced(self):
        self.assertEqual(
            normalize_doc_url("https://www.ibm.com/docs/en/cloud-paks/z-mod/2023.4?topic=baz"),
            "https://www.ibm.com/docs/en/cloud-paks/z-mod/latest?topic=baz",
        )

    def test_fragment_stripped(self):
        self.assertEqual(
            normalize_doc_url("https://www.ibm.com/docs/en/zos/2.5.0?topic=foo#section-2"),
            "https://www.ibm.com/docs/en/zos/latest?topic=foo",
        )

    def test_no_version_unchanged_except_fragment(self):
        self.assertEqual(
            normalize_doc_url("https://www.ibm.com/docs/en/z-devops-guide?topic=foo#bar"),
            "https://www.ibm.com/docs/en/z-devops-guide?topic=foo",
        )

    def test_non_docs_url_unchanged(self):
        self.assertEqual(
            normalize_doc_url("https://example.com/news/2024/03/21/article"),
            "https://example.com/news/2024/03/21/article",
        )

    def test_id_path_not_treated_as_version(self):
        self.assertEqual(
            normalize_doc_url("https://www.ibm.com/support/pages/node/6415115"),
            "https://www.ibm.com/support/pages/node/6415115",
        )

    def test_empty_and_none_safe(self):
        self.assertEqual(normalize_doc_url(""), "")
        self.assertIsNone(normalize_doc_url(None))


def _query(qid, relevant, norm_gold_urls=None):
    md = {}
    if norm_gold_urls is not None:
        md["norm-gold-urls"] = norm_gold_urls
    return SearchQueries.Query(
        default_query_template,
        _id=qid,
        relevant=relevant,
        metadata=md,
        text="q",
    )


def _result(query, passages):
    """Build a SearchResult with the given passages (list of dicts) attached.

    Dict shape: {id, score, metadata: {url: ...}}.
    """
    sr = SearchResult.__new__(SearchResult)
    sr.retrieved_passages = []
    sr.rouge_scorer = None
    sr.question = query
    for p in passages:
        sr.retrieved_passages.append(SearchResult.SearchDatum(p))
    return sr


def _eval_args(match_by="id"):
    a = EvaluationArguments(ranks="1,3", eval_measure="match,mrr,ndcg", match_by=match_by)
    return a


class TestUrlMatchEvaluation(unittest.TestCase):
    """Single 3-query scenario covering: url match hit, cross-version hit via
    normalization, and fallback to id-match when chunk lacks a url.
    """

    def setUp(self):
        # q1: gold doc id 'A' lives at zos/2.5.0; retrieval returns a chunk of
        # *different* doc id 'B' at the SAME url at rank 1. Id-match misses,
        # url-match hits.
        self.q1 = _query(
            "q1",
            relevant=["A"],
            norm_gold_urls=["https://www.ibm.com/docs/en/zos/latest?topic=foo"],
        )
        self.r1 = _result(self.q1, [
            {"id": "B-0-0", "score": 0.9, "metadata": {"url": "https://www.ibm.com/docs/en/zos/3.1.0?topic=foo"}},
            {"id": "C-0-0", "score": 0.8, "metadata": {"url": "https://example.com/x"}},
            {"id": "D-0-0", "score": 0.7, "metadata": {"url": "https://example.com/y"}},
        ])

        # q2: gold has cross-version urls; retrieval returns a chunk pointing to
        # a different version of the same page. Url-match hits via normalization.
        self.q2 = _query(
            "q2",
            relevant=["X"],
            norm_gold_urls=["https://www.ibm.com/docs/en/db2-for-zos/latest?topic=bar"],
        )
        self.r2 = _result(self.q2, [
            {"id": "Y-0-0", "score": 0.95, "metadata": {"url": "https://www.ibm.com/docs/en/db2-for-zos/12?topic=bar"}},
            {"id": "Z-0-0", "score": 0.5, "metadata": {"url": "https://example.com/z"}},
        ])

        # q3: chunk lacks metadata.url -> url mode falls back to id match.
        self.q3 = _query(
            "q3",
            relevant=["G"],
            norm_gold_urls=["https://www.ibm.com/docs/en/zos/latest?topic=baz"],
        )
        self.r3 = _result(self.q3, [
            {"id": "G-0-0", "score": 0.99},  # no metadata at all -> id-match wins
            {"id": "H-0-0", "score": 0.5, "metadata": {"url": "https://example.com/h"}},
        ])

        self.queries = [self.q1, self.q2, self.q3]
        self.system = [self.r1, self.r2, self.r3]

    def _run(self, match_by):
        engine = EvaluationEngine(_eval_args(match_by=match_by))
        return engine.compute_score(
            self.queries, self.system, query_template=default_query_template
        )

    def test_id_mode_misses_cross_id_url_hits(self):
        out = self._run("id")
        # q1 retrieves doc B (not gold A) at rank 1 -> miss
        # q2 retrieves doc Y (not gold X) -> miss
        # q3 retrieves doc G (gold) at rank 1 -> hit
        self.assertAlmostEqual(out.match[1], 1 / 3, places=4)
        self.assertAlmostEqual(out.match[3], 1 / 3, places=4)

    def test_url_mode_picks_up_cross_id_and_cross_version_hits(self):
        out = self._run("url")
        # All 3 queries now hit at rank 1: q1 via url, q2 via normalized url, q3 via id-fallback
        self.assertAlmostEqual(out.match[1], 1.0, places=4)
        self.assertAlmostEqual(out.match[3], 1.0, places=4)
        self.assertAlmostEqual(out.mrr[1], 1.0, places=4)
        self.assertAlmostEqual(out.ndcg[1], 1.0, places=4)

    def test_default_match_by_is_id(self):
        # Sanity check: omitting match_by behaves the same as match_by='id'.
        engine_default = EvaluationEngine(EvaluationArguments(ranks="1,3", eval_measure="match"))
        out = engine_default.compute_score(
            self.queries, self.system, query_template=default_query_template
        )
        self.assertAlmostEqual(out.match[1], 1 / 3, places=4)


if __name__ == "__main__":
    unittest.main()

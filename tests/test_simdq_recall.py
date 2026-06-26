"""T5 - SciFact recall regression: ingest + R0..R5 search, gate on baseline JSON.

Skips when the baseline contains nulls (the dormant CI gate). Populated by
running the test once with the --update-baseline CLI flag (registered in
tests/conftest.py):

    pytest tests/test_simdq_recall.py -m slow --update-baseline

After that run, tests/fixtures/simdq_scifact_baseline.json is rewritten with
measured values; commit it together with the kernel/encoder change that
justified the move.

The measured numbers are NDCG@10 and the codebase's `match@100` (which is
"any-relevant-found-at-K", an approximation of Recall@100 when most BEIR
qrels contain 1-2 relevant docs per query — the SciFact dataset fits that
profile). The gate is self-consistent: baseline + measured come from the
same scorer.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pytest

# Lazy imports — the whole module skips on environments lacking deps.
pytest.importorskip("sentence_transformers")
pytest.importorskip("datasets")

from datasets import load_dataset  # noqa: E402

from docuverse.engines.data_template import default_query_template  # noqa: E402
from docuverse.engines.search_queries import SearchQueries  # noqa: E402
from docuverse.engines.search_engine_config_params import (  # noqa: E402
    RetrievalArguments, EvaluationArguments,
)
from docuverse.utils.retrievers import create_retrieval_engine  # noqa: E402
from docuverse.utils.evaluator import EvaluationEngine  # noqa: E402

BASELINE_PATH = Path(__file__).parent / "fixtures" / "simdq_scifact_baseline.json"
ENCODER = "ibm-granite/granite-embedding-311m-multilingual-r2"

RECIPES = [
    ("R0", dict(simdq_family="hamming",    simdq_b=1, simdq_projection="identity",
                simdq_d=None, simdq_store_floats=True,  simdq_rescore_alpha=10)),
    ("R1", dict(simdq_family="asymmetric", simdq_b=1, simdq_projection="identity",
                simdq_d=None, simdq_store_floats=False, simdq_rescore_alpha=1)),
    ("R2", dict(simdq_family="asymmetric", simdq_b=1, simdq_projection="identity",
                simdq_d=None, simdq_store_floats=True,  simdq_rescore_alpha=10)),
    ("R3", dict(simdq_family="asymmetric", simdq_b=2, simdq_projection="identity",
                simdq_d=None, simdq_store_floats=False, simdq_rescore_alpha=1)),
    ("R4", dict(simdq_family="asymmetric", simdq_b=2, simdq_projection="random_orthogonal",
                simdq_d=384, simdq_store_floats=False, simdq_rescore_alpha=1)),
    ("R5", dict(simdq_family="asymmetric", simdq_b=4, simdq_projection="random_orthogonal",
                simdq_d=384, simdq_store_floats=False, simdq_rescore_alpha=1)),
    # Standardization A/B pairs (simdq_standardize=True). R6 mirrors R3 to isolate
    # the affine fix on the naked-sign identity path; R7 mirrors R5 to test it
    # combined with the random rotation. See docuverse/.../simdq/standardize.py.
    ("R6", dict(simdq_family="asymmetric", simdq_b=2, simdq_projection="identity",
                simdq_d=None, simdq_store_floats=False, simdq_rescore_alpha=1,
                simdq_standardize=True)),
    ("R7", dict(simdq_family="asymmetric", simdq_b=4, simdq_projection="random_orthogonal",
                simdq_d=384, simdq_store_floats=False, simdq_rescore_alpha=1,
                simdq_standardize=True)),
    # ITQ (learned_orthogonal) A/B pairs. R8 mirrors R6 (full dim, b=2) to ask
    # "does a learned rotation beat no rotation?"; R9 mirrors R7 (d=D/2, b=4) to
    # ask "does a learned rotation beat a random one?". Both standardize first.
    ("R8", dict(simdq_family="asymmetric", simdq_b=2, simdq_projection="learned_orthogonal",
                simdq_d=None, simdq_store_floats=False, simdq_rescore_alpha=1,
                simdq_standardize=True)),
    ("R9", dict(simdq_family="asymmetric", simdq_b=4, simdq_projection="learned_orthogonal",
                simdq_d=384, simdq_store_floats=False, simdq_rescore_alpha=1,
                simdq_standardize=True)),
]


def _load_baseline():
    return json.loads(BASELINE_PATH.read_text())


def _save_baseline(data):
    BASELINE_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _baseline_complete(b):
    return all(b["recipes"][rid]["ndcg10"] is not None and
               b["recipes"][rid]["recall100"] is not None
               for rid, _ in RECIPES)


class _ListCorpus:
    """Minimal corpus-like wrapper exposing len/getitem to engine.ingest."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


@pytest.fixture(scope="module")
def scifact():
    """Load (passages, queries-with-relevant) from BeIR/scifact.

    Restricts queries to the test split (those with at least one qrel).
    """
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")

    passages = [{"id": str(r["_id"]), "text": r["text"], "title": r.get("title", "")}
                for r in corpus]

    rel_map: dict[str, list[str]] = {}
    for r in qrels:
        rel_map.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))

    judged_qids = set(rel_map.keys())
    queries = []
    for r in queries_ds:
        qid = str(r["_id"])
        if qid not in judged_qids:
            continue
        queries.append(SearchQueries.Query(
            template=default_query_template,
            id=qid,
            text=r["text"],
            relevant=rel_map[qid],
        ))
    return passages, queries


def _make_args(td: Path, recipe_id: str, overrides: dict) -> RetrievalArguments:
    args = RetrievalArguments()
    args.db_engine = "simdq"
    args.model_name = ENCODER
    args.index_name = f"scifact_{recipe_id}"
    args.project_dir = str(td)
    args.top_k = 100  # required for recall@100 measurement
    args.ingestion_batch_size = 64
    args.max_text_size = 512
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _eval_args() -> EvaluationArguments:
    ev = EvaluationArguments()
    ev.ranks = "10,100"
    ev.eval_measure = "match,ndcg"
    ev.__post_init__()
    return ev


def _run_recipe(recipe_id, overrides, passages, queries, project_dir: Path):
    args = _make_args(project_dir, recipe_id, overrides)
    engine = create_retrieval_engine(args)
    engine.ingest(_ListCorpus(passages), update=False)
    # RetrievalEngine.search handles one query at a time (the SearchEngine
    # orchestration layer fans these out via parallel_process); call it per query.
    output = [engine.search(q) for q in queries]
    scorer = EvaluationEngine(_eval_args())
    # model_name is only a display label here (get_output_name lives on the
    # SearchEngine orchestration layer, not the retrieval engine).
    eval_out = scorer.compute_score(queries, output,
                                    model_name=f"{ENCODER}:{recipe_id}")
    return {
        "ndcg10": float(eval_out.ndcg[10]),
        "recall100": float(eval_out.match[100]),
    }


@pytest.mark.slow
def test_scifact_recall_baseline(scifact, tmp_path_factory, request):
    passages, queries = scifact
    baseline = _load_baseline()
    update = request.config.getoption("--update-baseline")
    project_dir = tmp_path_factory.mktemp("scifact_simdq")

    if not update and not _baseline_complete(baseline):
        pytest.skip(
            "SciFact baseline contains null values. Populate with: "
            "pytest tests/test_simdq_recall.py -m slow --update-baseline"
        )

    measured = {}
    for rid, overrides in RECIPES:
        measured[rid] = _run_recipe(rid, overrides, passages, queries, project_dir)

    if update:
        baseline["recipes"] = measured
        baseline["scifact_revision"] = "BeIR/scifact"
        baseline["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        baseline["captured_on"] = platform.processor() or platform.machine()
        _save_baseline(baseline)
        pytest.skip("Baseline updated; re-run without --update-baseline to gate.")

    tol = baseline["tolerance"]
    failures = []
    for rid, _ in RECIPES:
        b = baseline["recipes"][rid]
        m = measured[rid]
        if abs(m["ndcg10"] - b["ndcg10"]) > tol["ndcg10"]:
            failures.append(
                f"{rid} ndcg10 {m['ndcg10']:.4f} vs baseline {b['ndcg10']:.4f} "
                f"(tol {tol['ndcg10']})"
            )
        if abs(m["recall100"] - b["recall100"]) > tol["recall100"]:
            failures.append(
                f"{rid} recall100 {m['recall100']:.4f} vs baseline {b['recall100']:.4f} "
                f"(tol {tol['recall100']})"
            )
    if failures:
        artifacts = Path("_artifacts")
        artifacts.mkdir(exist_ok=True)
        (artifacts / "scifact_measured.json").write_text(json.dumps(measured, indent=2))
        pytest.fail("Recall regression:\n  " + "\n  ".join(failures))

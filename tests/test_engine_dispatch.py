"""Table-driven dispatch tests for ``create_retrieval_engine``.

The dispatcher in ``docuverse/utils/retrievers.py`` is an if/elif ladder over
``db_engine`` strings. We don't want to instantiate real engines here (that
would require torch + pymilvus + elasticsearch + chromadb installed in CI),
so each test feeds a stub config and asserts the dispatcher reaches the
right branch — distinguishing "unknown engine" from "import failed" from
"engine instantiated".

This also locks in the trailing-space fix on ``'milvus_splade '`` (PR 1) by
asserting the bare string ``'milvus_splade'`` is recognized.
"""
from __future__ import annotations

import pytest

from docuverse.utils import retrievers


class _StubConfig(dict):
    """Minimal dict-with-attributes; the dispatcher only calls ``.get``."""

    def __getattr__(self, k):  # pragma: no cover - never used in dispatch
        return self.get(k)


KNOWN_NAMES = [
    "elastic-bm25",
    "es-bm25",
    "elastic-dense",
    "es-dense",
    "elastic-elser",
    "es-elser",
    "chromadb",
    "faiss",
    "milvus",
    "milvus-dense",
    "milvus_dense",
    "milvus-sparse",
    "milvus_sparse",
    "milvus-bm25",
    "milvus_bm25",
    "milvus-hybrid",
    "milvus_hybrid",
    "milvus-splade",
    "milvus_splade",  # locks the trailing-space fix from PR 1
    "lancedb",
    "lance",
    "lancedb-dense",
    "lancedb_dense",
    "lancedb-bm25",
    "lancedb_bm25",
    "lancedb-sparse",
    "lancedb_sparse",
    "lancedb-hybrid",
    "lancedb_hybrid",
]


@pytest.mark.parametrize("name", KNOWN_NAMES)
def test_known_db_engine_dispatches(name):
    """Every known name must reach a real branch.

    Three acceptable outcomes (all mean dispatch worked):
      1. Engine instantiated (optional dep installed + stub config sufficient).
      2. ``ImportError`` (optional dep missing — the dispatcher saw the name).
      3. ``TypeError`` / ``AttributeError`` from the engine constructor
         choking on our minimal stub config (it still got *to* the engine).

    What must NOT happen: the dispatcher returning ``None`` (unknown name)
    or raising ``NotImplementedError`` ("Unknown engine type:").
    """
    cfg = _StubConfig({"db_engine": name})
    try:
        engine = retrievers.create_retrieval_engine(cfg)
    except ImportError:
        return  # optional dep missing → dispatch reached the right branch
    except (TypeError, AttributeError, KeyError):
        return  # engine constructor needed more config → dispatch was correct
    except RuntimeError as e:
        # Several engine modules re-raise their ImportError as RuntimeError
        # ("You need to install pymilvus..."), and engine constructors raise
        # RuntimeError when given a stub config that doesn't include
        # ``index_name`` etc. Both mean dispatch worked.
        msg = str(e).lower()
        config_or_dep = any(
            tok in msg
            for tok in (
                "install",
                "import",
                "required",
                "index_name",
                "model_name",
                "config",
            )
        )
        if not config_or_dep:
            raise
        return
    except NotImplementedError as e:
        pytest.fail(f"{name!r} hit the 'Unknown engine type' branch: {e}")
    # If we got here, an engine was instantiated. That's also fine.
    if engine is None:
        pytest.fail(
            f"{name!r} returned None from create_retrieval_engine — "
            f"means the dispatcher had no matching branch (regression?)"
        )


def test_milvus_splade_no_trailing_space():
    """PR 1 fixed a typo where ``'milvus_splade '`` had a stray trailing space.

    Reading the source directly is the cleanest way to assert the fix can't
    silently come back; it's a one-character bug that's invisible to most
    other tests.
    """
    import inspect

    src = inspect.getsource(retrievers.create_retrieval_engine)
    assert "'milvus_splade '" not in src
    assert '"milvus_splade "' not in src
    assert "'milvus_splade'" in src or '"milvus_splade"' in src

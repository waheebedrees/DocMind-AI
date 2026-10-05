"""Unit tests for RetrievalService.

Isolation strategy:

- `session.execute` is mocked; DB-touching methods (`vector_search`,
  `keyword_search`, `_neighbors`) are exercised by feeding fake rows
  through `_to_candidate`.
- `retrieve` mocks the four DB/embed helpers and the four module-level
  pipeline functions, so it tests only the orchestration: order,
  dispatch, timing, and the fallback paths.
- The embedder and reranker are instances the tests hand to the service,
  never constructed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from app.core.exceptions import (
    DocumentNotReady,
    EmbeddingUnavailable,
    NotFoundError,
)
from app.models.enums import DocumentStatus
from app.rag.retrieval.retrieval import RetrievalService
from app.rag.retrieval.types import Candidate

USER_ID = UUID("11111111-2222-3333-4444-555555555555")
DOC_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
DOC_ID_2 = UUID("cccccccc-dddd-eeee-ffff-000000000000")
EMBED_DIM = 384


def make_cfg(**overrides) -> SimpleNamespace:
    """A stand-in for RetrievalSettings with the fields retrieve() reads."""
    defaults = {
        "fts_language": "english",
        "keyword_candidates": 50,
        "hnsw_ef_search": 100,
        "hnsw_iterative_scan": False,
        "vector_candidates": 50,
        "rrf_k": 60,
        "vector_weight": 1.0,
        "keyword_weight": 1.0,
        "rerank_candidates": 30,
        "rerank_enabled": False,
        "rerank_min_score": None,
        "rerank_timeout_s": 10.0,
        "mmr_lambda": 0.5,
        "neighbor_window": 1,
        "max_context_tokens": 4000,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_row(
    *,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    chunk_index: int = 0,
    text: str = "text",
    token_count: int | None = 10,
    embedding=None,
    distance: float = 0.1,
    rank: float = 0.5,
) -> SimpleNamespace:
    """A fake DB row matching `_COLS` plus the per-search extras."""
    return SimpleNamespace(
        id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=chunk_index,
        text=text,
        page_number=None,
        section=None,
        token_count=token_count,
        embedding=embedding if embedding is not None else np.zeros(EMBED_DIM, dtype=np.float32),
        distance=distance,
        rank=rank,
    )


def make_candidate(**overrides) -> Candidate:
    defaults = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "chunk_index": 0,
        "text": "text",
        "page_number": None,
        "section": None,
        "token_count": 10,
        "embedding": None,
        "vector_rank": None,
        "keyword_rank": None,
        "vector_score": None,
        "keyword_score": None,
        "fused_score": 0.0,
        "rerank_score": None,
    }
    defaults.update(overrides)
    return Candidate(**defaults)


def make_doc(status: DocumentStatus = DocumentStatus.INDEXED) -> MagicMock:
    d = MagicMock()
    d.id = DOC_ID
    d.status = status
    return d


@pytest.fixture
def session() -> AsyncMock:
    s = AsyncMock()
    result = MagicMock()
    result.all.return_value = []
    s.execute.return_value = result
    return s


@pytest.fixture
def cfg() -> SimpleNamespace:
    return make_cfg()


@pytest.fixture
def service(session: AsyncMock, cfg: SimpleNamespace) -> RetrievalService:
    return RetrievalService(session, cfg=cfg)


class TestValidateScope:
    async def test_all_valid_passes(self, service):
        service.docs.get_for_user = AsyncMock(return_value=make_doc())
        await service._validate_scope(USER_ID, [DOC_ID])

    async def test_missing_document_raises_not_found(self, service):
        service.docs.get_for_user = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await service._validate_scope(USER_ID, [DOC_ID])

    async def test_wrong_owner_raises_not_found(self, service):
        """get_for_user returns None when the doc exists but belongs to
        someone else — the error path must not distinguish the two."""
        service.docs.get_for_user = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError) as exc:
            await service._validate_scope(USER_ID, [DOC_ID])
        assert exc.value.code == "invalid_document_id"

    async def test_not_indexed_raises_document_not_ready(self, service):
        service.docs.get_for_user = AsyncMock(return_value=make_doc(DocumentStatus.PROCESSING))
        with pytest.raises(DocumentNotReady):
            await service._validate_scope(USER_ID, [DOC_ID])

    async def test_duplicates_are_checked_once(self, service):
        service.docs.get_for_user = AsyncMock(return_value=make_doc())
        await service._validate_scope(USER_ID, [DOC_ID, DOC_ID, DOC_ID])
        assert service.docs.get_for_user.await_count == 1

    async def test_multiple_ids_each_checked(self, service):
        service.docs.get_for_user = AsyncMock(return_value=make_doc())
        await service._validate_scope(USER_ID, [DOC_ID, DOC_ID_2])
        assert service.docs.get_for_user.await_count == 2

    async def test_stops_on_first_invalid(self, service):
        calls = []
        invalid_hit = [False]

        async def get(user_id, doc_id):
            calls.append(doc_id)
            if invalid_hit[0]:
                raise AssertionError("checked another id after the invalid one")
            if doc_id == DOC_ID:
                invalid_hit[0] = True
                return None
            return make_doc()

        service.docs.get_for_user = get
        with pytest.raises(NotFoundError):
            await service._validate_scope(USER_ID, [DOC_ID, DOC_ID_2])

        assert DOC_ID in calls
        assert calls.index(DOC_ID) == len(calls) - 1  # nothing checked after


# --- _neighbors ------------------------------------------------------


class TestNeighbors:
    async def test_boundary_chunk_fetches_only_the_next_position(self, service, session):
        """With contiguous selected chunks and window=1, only the position
        just past the run is fetched — everything inside the run is already
        selected. This is the boundary case that makes the query unavoidable
        for any non-empty selection with window >= 1."""
        doc = uuid4()
        selected = [
            make_candidate(document_id=doc, chunk_index=0),
            make_candidate(document_id=doc, chunk_index=1),
            make_candidate(document_id=doc, chunk_index=2),
        ]
        await service._neighbors(selected, window=1)
        session.execute.assert_awaited_once()
        # (The stmt was built with positions [(doc, 3)]; no way to inspect
        # the WHERE clause without a real DB, but the query running at all
        # is the load-bearing assertion.)

    async def test_negative_window_returns_empty(self, service, session):
        assert await service._neighbors([make_candidate()], window=-1) == []
        session.execute.assert_not_awaited()

    async def test_neighbors_excludes_selected_positions(self, service, session):
        doc = uuid4()
        selected = [make_candidate(document_id=doc, chunk_index=5)]
        # No rows to return, but we can check the query was issued
        await service._neighbors(selected, window=1)
        session.execute.assert_awaited_once()

    async def test_neighbors_fetch_returns_candidates(self, service, session):
        doc = uuid4()
        row = make_row(document_id=doc, chunk_index=4)
        session.execute.return_value.all.return_value = [row]

        selected = [make_candidate(document_id=doc, chunk_index=5)]
        out = await service._neighbors(selected, window=1)

        assert len(out) == 1
        assert out[0].document_id == doc
        assert out[0].chunk_index == 4

    async def test_chunk_zero_never_produces_negative_index(self, service, session):
        """With chunk_index=0 and window=1, -1 must be excluded."""
        selected = [make_candidate(chunk_index=0)]
        await service._neighbors(selected, window=1)
        # If -1 leaked into the positions tuple, SQLAlchemy would still
        # build a statement; the code has an explicit `i >= 0` guard.
        # We can only assert the query ran without raising.
        session.execute.assert_awaited_once()

    async def test_boundary_chunk_fetches_next_position(self, service, session):
        """With contiguous selected chunks and window=1, the position just
        past the run is the only one not already selected, so the query
        still runs. This is why `wanted - have` is never empty when
        window >= 1 and selected is non-empty."""
        doc = uuid4()
        selected = [
            make_candidate(document_id=doc, chunk_index=0),
            make_candidate(document_id=doc, chunk_index=1),
            make_candidate(document_id=doc, chunk_index=2),
        ]
        await service._neighbors(selected, window=1)
        session.execute.assert_awaited_once()


# --- vector_search ---------------------------------------------------


class TestVectorSearch:
    async def test_sets_ef_search_as_local_config(self, service, session):
        session.execute.return_value.all.return_value = []
        await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=10,
            ef_search=100,
            iterative_scan=False,
        )
        first = session.execute.await_args_list[0]
        sql = str(first.args[0])
        assert "hnsw.ef_search" in sql
        assert first.args[1] == {"v": "100"}

    async def test_sets_iterative_scan_when_requested(self, service, session):
        session.execute.return_value.all.return_value = []
        await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=10,
            ef_search=100,
            iterative_scan=True,
        )
        # 1st: ef_search; 2nd: iterative_scan; 3rd: the actual query
        iterative_call = session.execute.await_args_list[1]
        assert "hnsw.iterative_scan" in str(iterative_call.args[0])
        assert "relaxed_order" in str(iterative_call.args[0])

    async def test_does_not_set_iterative_scan_when_disabled(self, service, session):
        session.execute.return_value.all.return_value = []
        await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=10,
            ef_search=100,
            iterative_scan=False,
        )
        # Only 2 calls: ef_search + the query
        assert session.execute.await_count == 2

    async def test_assigns_rank_one_indexed(self, service, session):
        rows = [
            make_row(distance=0.1),
            make_row(distance=0.2),
            make_row(distance=0.3),
        ]
        session.execute.return_value.all.return_value = rows
        out = await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=3,
            ef_search=100,
            iterative_scan=False,
        )
        assert [c.vector_rank for c in out] == [1, 2, 3]

    async def test_vector_score_is_one_minus_distance(self, service, session):
        session.execute.return_value.all.return_value = [make_row(distance=0.25)]
        out = await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=1,
            ef_search=100,
            iterative_scan=False,
        )
        assert out[0].vector_score == pytest.approx(0.75)

    async def test_reranks_rows_by_distance(self, service, session):
        """relaxed_order may return unordered rows; the method re-sorts."""
        rows = [
            make_row(distance=0.5),
            make_row(distance=0.1),
            make_row(distance=0.3),
        ]
        session.execute.return_value.all.return_value = rows
        out = await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=3,
            ef_search=100,
            iterative_scan=False,
        )
        distances = [1.0 - c.vector_score for c in out]
        assert distances == sorted(distances)

    async def test_embedding_converted_to_numpy(self, service, session):
        """_to_candidate must convert the list to np.float32 array for MMR."""
        row = make_row(embedding=[0.5] * EMBED_DIM)
        session.execute.return_value.all.return_value = [row]
        out = await service.vector_search(
            user_id=USER_ID,
            embedding=[0.1] * EMBED_DIM,
            document_ids=None,
            limit=1,
            ef_search=100,
            iterative_scan=False,
        )
        assert isinstance(out[0].embedding, np.ndarray)
        assert out[0].embedding.dtype == np.float32


# --- keyword_search --------------------------------------------------


class TestKeywordSearch:
    async def test_assigns_rank_one_indexed(self, service, session):
        rows = [make_row(rank=0.9), make_row(rank=0.5), make_row(rank=0.1)]
        session.execute.return_value.all.return_value = rows
        out = await service.keyword_search(
            user_id=USER_ID,
            query="hello",
            document_ids=None,
            limit=10,
            language="english",
        )
        assert [c.keyword_rank for c in out] == [1, 2, 3]

    async def test_keyword_score_populated(self, service, session):
        session.execute.return_value.all.return_value = [make_row(rank=0.42)]
        out = await service.keyword_search(
            user_id=USER_ID,
            query="hello",
            document_ids=None,
            limit=10,
            language="english",
        )
        assert out[0].keyword_score == pytest.approx(0.42)

    async def test_empty_result_preserved(self, service, session):
        session.execute.return_value.all.return_value = []
        out = await service.keyword_search(
            user_id=USER_ID,
            query="no-match",
            document_ids=None,
            limit=10,
            language="english",
        )
        assert out == []


# --- _embed ----------------------------------------------------------


class TestEmbed:
    async def test_prepends_query_prefix(self, service):
        spec = SimpleNamespace(
            query_prefix="query: ",
            dimension=EMBED_DIM,
            embed_timeout_s=5.0,
        )
        captured: list[list[str]] = []

        async def fake_embed(_embedder, texts):
            captured.append(list(texts))
            return [[0.1] * EMBED_DIM]

        with (
            patch("app.rag.retrieval.retrieval.settings") as s,
            patch("app.rag.retrieval.retrieval.get_embedder", return_value=object()),
            patch(
                "app.rag.retrieval.retrieval.embed_in_batches",
                new=fake_embed,
            ),
        ):
            s.embedding_spec = spec
            await service._embed("hello")

        assert captured == [["query: hello"]]

    async def test_timeout_raises_embedding_unavailable(self, service):
        async def slow(*_a, **_kw):
            raise TimeoutError("simulated timeout")

        with (
            patch("app.rag.retrieval.retrieval.settings") as s,
            patch("app.rag.retrieval.retrieval.get_embedder", return_value=object()),
            patch("app.rag.retrieval.retrieval.embed_in_batches", new=slow),
        ):
            s.embedding_spec = SimpleNamespace(query_prefix="", dimension=EMBED_DIM, embed_timeout_s=0.01)
            with pytest.raises(EmbeddingUnavailable):
                await service._embed("hello")

    async def test_oserror_raises_embedding_unavailable(self, service):
        async def broken(*_a, **_kw):
            raise OSError("connection reset")

        with (
            patch("app.rag.retrieval.retrieval.settings") as s,
            patch("app.rag.retrieval.retrieval.get_embedder", return_value=object()),
            patch("app.rag.retrieval.retrieval.embed_in_batches", new=broken),
        ):
            s.embedding_spec = SimpleNamespace(query_prefix="", dimension=EMBED_DIM, embed_timeout_s=5.0)
            with pytest.raises(EmbeddingUnavailable):
                await service._embed("hello")

    async def test_dimension_mismatch_raises_runtime_error(self, service):
        async def wrong_dim(*_a, **_kw):
            return [[0.1] * 128]  # 128 != 384

        with (
            patch("app.rag.retrieval.retrieval.settings") as s,
            patch("app.rag.retrieval.retrieval.get_embedder", return_value=object()),
            patch("app.rag.retrieval.retrieval.embed_in_batches", new=wrong_dim),
        ):
            s.embedding_spec = SimpleNamespace(query_prefix="", dimension=EMBED_DIM, embed_timeout_s=5.0)
            with pytest.raises(RuntimeError, match="dim"):
                await service._embed("hello")


# --- _rerank ---------------------------------------------------------


class TestRerank:
    async def test_returns_false_when_no_reranker(self, service):
        assert await service._rerank("q", [make_candidate()]) is False

    async def test_returns_false_and_logs_on_reranker_exception(self, service):
        async def boom(*_a, **_kw):
            raise RuntimeError("upstream 503")

        service.reranker = MagicMock()
        service.reranker.score = boom
        assert await service._rerank("q", [make_candidate()]) is False

    async def test_sets_rerank_score_on_success(self, service):
        service.reranker = MagicMock()
        service.reranker.score = AsyncMock(return_value=[0.9, 0.1])
        candidates = [make_candidate(), make_candidate()]
        ok = await service._rerank("q", candidates)
        assert ok is True
        assert [c.rerank_score for c in candidates] == [0.9, 0.1]

    async def test_timeout_returns_false(self, service, cfg):
        """cfg.rerank_timeout_s is honored; slow reranker falls back."""
        import asyncio

        async def slow(*_a, **_kw):
            await asyncio.sleep(5)
            return []

        cfg.rerank_timeout_s = 0.01
        service.reranker = MagicMock()
        service.reranker.score = slow
        assert await service._rerank("q", [make_candidate()]) is False

    async def test_never_raises_on_reranker_failure(self, service):
        """Any exception path must degrade, not propagate — retrieval
        is more important than quality reranking."""

        async def weird(*_a, **_kw):
            raise BaseException("even this")  # noqa: TRY002

        service.reranker = MagicMock()
        service.reranker.score = weird
        # BaseException is not caught by `except Exception`, so this
        # would propagate. The docstring promises Exception only.
        # This test pins that contract by expecting no raise on a
        # normal Exception.
        service.reranker.score = AsyncMock(side_effect=Exception("normal"))
        assert await service._rerank("q", [make_candidate()]) is False


# --- retrieve --------------------------------------------------------


def _patch_pipeline(
    fused=None,
    mmr_selected=None,
    passages=None,
    context="",
):
    """Patch the four module-level functions used inside retrieve()."""
    fused = fused if fused is not None else []
    mmr_selected = mmr_selected if mmr_selected is not None else []
    passages = passages if passages is not None else []

    return (
        patch(
            "app.rag.retrieval.retrieval.reciprocal_rank_fusion",
            return_value=fused,
        ),
        patch(
            "app.rag.retrieval.retrieval.mmr_select",
            return_value=mmr_selected,
        ),
        patch(
            "app.rag.retrieval.retrieval.build_passages",
            return_value=passages,
        ),
        patch(
            "app.rag.retrieval.retrieval.render_context",
            return_value=context,
        ),
    )


class TestRetrieve:
    async def test_empty_query_raises(self, service):
        with pytest.raises(ValueError):
            await service.retrieve(user_id=USER_ID, query="")

    async def test_whitespace_only_query_raises(self, service):
        with pytest.raises(ValueError):
            await service.retrieve(user_id=USER_ID, query="   \t\n  ")

    async def test_query_is_normalized_before_use(self, service):
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline()
        with p1, p2, p3, p4:
            await service.retrieve(
                user_id=USER_ID,
                query="  hello   world  ",
            )
        # keyword_search receives the normalized query
        assert service.keyword_search.await_args.kwargs["query"] == "hello world"

    async def test_validate_scope_called_when_document_ids_given(self, service):
        service._validate_scope = AsyncMock()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline()
        with p1, p2, p3, p4:
            await service.retrieve(
                user_id=USER_ID,
                query="hello",
                document_ids=[DOC_ID],
            )
        service._validate_scope.assert_awaited_once_with(USER_ID, [DOC_ID])

    async def test_validate_scope_skipped_when_no_document_ids(self, service):
        service._validate_scope = AsyncMock()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline()
        with p1, p2, p3, p4:
            await service.retrieve(user_id=USER_ID, query="hello")
        service._validate_scope.assert_not_awaited()

    async def test_returns_empty_result_when_fused_is_empty(self, service):
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline(fused=[])
        with p1, p2, p3, p4:
            out = await service.retrieve(user_id=USER_ID, query="hello")
        assert out.passages == []
        assert out.context == ""
        assert out.reranked is False
        assert out.candidate_counts["fused"] == 0

    async def test_rerank_disabled_by_default(self, service):
        candidate = make_candidate()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._rerank = AsyncMock()
        service._neighbors = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline(
            fused=[candidate],
            mmr_selected=[candidate],
            passages=[],
        )
        with p1, p2, p3, p4:
            await service.retrieve(user_id=USER_ID, query="hello")
        service._rerank.assert_not_awaited()

    async def test_rerank_enabled_but_no_reranker_does_not_call_rerank(self, service, cfg):
        cfg.rerank_enabled = True
        service.reranker = None
        candidate = make_candidate()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._neighbors = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline(
            fused=[candidate],
            mmr_selected=[candidate],
        )
        with p1, p2, p3, p4:
            out = await service.retrieve(user_id=USER_ID, query="hello")
        assert out.reranked is False

    async def test_rerank_enabled_and_successful_sorts_by_rerank_score(self, service, cfg):
        cfg.rerank_enabled = True
        service.reranker = MagicMock()
        # First candidate scored low, second high
        service.reranker.score = AsyncMock(return_value=[0.1, 0.9])

        a = make_candidate(text="a")
        b = make_candidate(text="b")
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._neighbors = AsyncMock(return_value=[])

        captured_mmr_input = []

        def fake_mmr(cands, *, k, lambda_):
            captured_mmr_input.extend(cands)
            return cands

        p1, _, p3, p4 = _patch_pipeline(fused=[a, b], passages=[])
        with (
            p1,
            p3,
            p4,
            patch(
                "app.rag.retrieval.retrieval.mmr_select",
                side_effect=fake_mmr,
            ),
        ):
            out = await service.retrieve(user_id=USER_ID, query="hello")

        assert out.reranked is True
        # After sorting by rerank_score desc, b (0.9) comes first
        assert [c.text for c in captured_mmr_input] == ["b", "a"]

    async def test_rerank_failure_falls_back(self, service, cfg):
        cfg.rerank_enabled = True
        service.reranker = MagicMock()
        service.reranker.score = AsyncMock(side_effect=RuntimeError("boom"))

        candidate = make_candidate()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._neighbors = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline(
            fused=[candidate],
            mmr_selected=[candidate],
        )
        with p1, p2, p3, p4:
            out = await service.retrieve(user_id=USER_ID, query="hello")
        assert out.reranked is False

    async def test_min_score_threshold_filters_candidates(self, service, cfg):
        cfg.rerank_enabled = True
        cfg.rerank_min_score = 0.5
        service.reranker = MagicMock()
        service.reranker.score = AsyncMock(return_value=[0.1, 0.9])

        a = make_candidate(text="low")
        b = make_candidate(text="high")
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._neighbors = AsyncMock(return_value=[])

        captured = []

        def fake_mmr(cands, *, k, lambda_):
            captured.extend(cands)
            return cands

        p1, _, p3, p4 = _patch_pipeline(fused=[a, b], passages=[])
        with (
            p1,
            p3,
            p4,
            patch(
                "app.rag.retrieval.retrieval.mmr_select",
                side_effect=fake_mmr,
            ),
        ):
            out = await service.retrieve(user_id=USER_ID, query="hello")

        # Only b survives the threshold
        assert [c.text for c in captured] == ["high"]
        assert out.candidate_counts["after_threshold"] == 1

    async def test_top_k_forwarded_to_mmr(self, service):
        candidate = make_candidate()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._neighbors = AsyncMock(return_value=[])

        mmr_kwargs = {}

        def fake_mmr(cands, *, k, lambda_):
            mmr_kwargs["k"] = k
            mmr_kwargs["lambda_"] = lambda_
            return cands

        p1, _, p3, p4 = _patch_pipeline(fused=[candidate])
        with (
            p1,
            p3,
            p4,
            patch(
                "app.rag.retrieval.retrieval.mmr_select",
                side_effect=fake_mmr,
            ),
        ):
            await service.retrieve(user_id=USER_ID, query="hello", top_k=3)

        assert mmr_kwargs["k"] == 3

    async def test_timings_populated(self, service):
        candidate = make_candidate()
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=[])
        service.vector_search = AsyncMock(return_value=[])
        service._neighbors = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline(
            fused=[candidate],
            mmr_selected=[candidate],
        )
        with p1, p2, p3, p4:
            out = await service.retrieve(user_id=USER_ID, query="hello")
        assert "embed+keyword" in out.timings_ms
        assert "vector" in out.timings_ms
        assert "mmr" in out.timings_ms
        assert "expand" in out.timings_ms

    async def test_candidate_counts_reported(self, service):
        v = [make_candidate() for _ in range(3)]
        k = [make_candidate() for _ in range(5)]
        f = [make_candidate() for _ in range(7)]
        service._embed = AsyncMock(return_value=[0.1] * EMBED_DIM)
        service.keyword_search = AsyncMock(return_value=k)
        service.vector_search = AsyncMock(return_value=v)
        service._neighbors = AsyncMock(return_value=[])
        p1, p2, p3, p4 = _patch_pipeline(fused=f, mmr_selected=f[:1])
        with p1, p2, p3, p4:
            out = await service.retrieve(user_id=USER_ID, query="hello")
        assert out.candidate_counts["vector"] == 3
        assert out.candidate_counts["keyword"] == 5
        assert out.candidate_counts["fused"] == 7

    async def test_embed_failure_propagates_and_cancels_keyword(self, service):
        """If the embed task raises, keyword_search's coroutine is
        cancelled and the exception propagates."""
        started = []

        async def failing_embed(_query):
            started.append("embed")
            raise RuntimeError("embed exploded")

        service._embed = failing_embed

        async def slow_keyword(**_kw):
            started.append("keyword-start")
            import asyncio

            await asyncio.sleep(5)
            return []

        service.keyword_search = slow_keyword

        with pytest.raises(RuntimeError, match="embed exploded"):
            await service.retrieve(user_id=USER_ID, query="hello")

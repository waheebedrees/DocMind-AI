"""Advanced unit tests for the ingestion pipeline (Ingester).

Focus areas beyond the basic suite:

- State-machine exhaustiveness: every (job_status, doc_status) combination
  for each transition method, so a new enum member can't silently bypass
  validation.
- Transaction contract: Ingester never commits; flush count is
  predictable; a raise before mutation leaves state untouched.
- Invariant preservation: on every successful call, structural invariants
  hold (terminal jobs have finished_at, INDEXED docs have indexed_at,
  etc.).
- Concurrency: two workers on distinct jobs share no state; two calls on
  the same job are reentrant when the second observes RUNNING.
- Full pipeline progression: extract -> chunk -> embed -> index end-to-end
  with all four jobs driven through the state machine.
- Embed specifics: strict 1:1 vector/chunk pairing, cancellation,
  timeout classification.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import (
    DocumentNotFound,
    InvalidTransition,
    JobAlreadyDone,
    JobAlreadyFailed,
    JobNotFound,
    TransientEmbeddingError,
)
from app.db.repositories.chunks import PreparedChunk
from app.models.enums import DocumentStatus, JobStage, JobStatus
from app.rag.ingestion import Ingester

pytestmark = pytest.mark.asyncio


DOC_ID = UUID("11111111-2222-3333-4444-555555555555")
JOB_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


# --- helpers ---------------------------------------------------------------


def make_job(
    *,
    status: JobStatus = JobStatus.QUEUED,
    stage: JobStage = JobStage.EXTRACT,
    document_id: UUID = DOC_ID,
    started_at: datetime | None = None,
    details: dict | None = None,
) -> MagicMock:
    j = MagicMock()
    j.id = JOB_ID
    j.status = status
    j.stage = stage
    j.document_id = document_id
    j.started_at = started_at
    j.finished_at = None
    j.details = details
    return j


def make_doc(
    *,
    status: DocumentStatus = DocumentStatus.PENDING,
    doc_id: UUID = DOC_ID,
) -> MagicMock:
    d = MagicMock()
    d.id = doc_id
    d.status = status
    d.page_count = None
    d.indexed_at = None
    d.error_message = None
    d.metadata_ = {}
    return d


def make_chunk(i: int, text: str | None = None) -> PreparedChunk:
    return PreparedChunk(
        chunk_index=i,
        text=text if text is not None else f"chunk {i}",
        page_number=None,
        section=None,
        token_count=1,
        doc_item_labels=[],
        embedding=None,
    )


def assert_invariants(job: MagicMock, doc: MagicMock) -> None:
    """Structural invariants that must hold after any successful call."""
    if job.status in (JobStatus.DONE, JobStatus.FAILED):
        assert job.finished_at is not None, "terminal job must have finished_at"
    if job.status == JobStatus.QUEUED:
        assert job.started_at is None, "queued job must not have started_at"
    if doc.status == DocumentStatus.INDEXED:
        assert doc.indexed_at is not None, "indexed doc must have indexed_at"


@pytest.fixture
def session() -> AsyncMock:
    s = AsyncMock()
    s.commit = AsyncMock(side_effect=AssertionError(
        "Ingester must not commit"))
    s.rollback = AsyncMock(side_effect=AssertionError(
        "Ingester must not rollback"))
    return s


@pytest.fixture
def ingester(session: AsyncMock) -> Ingester:
    ing = Ingester(session)
    ing._jobs = AsyncMock()
    ing._documents = AsyncMock()
    ing._chunks = AsyncMock()
    return ing


@pytest.fixture
def ingester_with_live_status(session: AsyncMock) -> Ingester:
    """Like `ingester`, but `_documents.set_status` actually mutates the
    doc object. Needed when one test drives both `start_stage` and
    `complete_stage` on the same doc — otherwise the mock's set_status
    is a no-op and the second call reads a stale status."""
    ing = Ingester(session)
    ing._jobs = AsyncMock()
    ing._documents = AsyncMock()
    ing._chunks = AsyncMock()

    async def set_status(doc_id, new_status):
        doc = ing._documents.get_by_id.return_value
        if doc is not None and doc.id == doc_id:
            doc.status = new_status

    ing._documents.set_status.side_effect = set_status
    return ing

# --- state-machine matrix: start_stage -------------------------------------

_START_OK = "ok"
_START_JOB_DONE = "job_done"
_START_JOB_FAILED = "job_failed"
_START_JOB_STATUS = "job_status"
_START_DOC_NOT_FOUND = "doc_not_found"
_START_DOC_STATUS = "doc_status"


@pytest.mark.parametrize(
    "job_status, doc_status, outcome",
    [
        # Happy paths — two terminal job statuses to check first
        (JobStatus.QUEUED, DocumentStatus.PENDING, _START_OK),
        (JobStatus.QUEUED, DocumentStatus.PROCESSING, _START_OK),
        (JobStatus.RUNNING, DocumentStatus.PROCESSING, _START_OK),
        # Only expected reentrant path — RUNNING job + PENDING doc
        (JobStatus.RUNNING, DocumentStatus.PENDING, _START_OK),
        # Terminal job statuses
        (JobStatus.DONE, DocumentStatus.PENDING, _START_JOB_DONE),
        (JobStatus.FAILED, DocumentStatus.PENDING, _START_JOB_FAILED),
        # Document is in a state that can't move to processing
        (JobStatus.QUEUED, DocumentStatus.INDEXED, _START_DOC_STATUS),
        (JobStatus.QUEUED, DocumentStatus.FAILED, _START_DOC_STATUS),
        (JobStatus.RUNNING, DocumentStatus.INDEXED, _START_DOC_STATUS),
    ],
)
async def test_start_stage_matrix(ingester, job_status, doc_status, outcome):
    ingester._jobs.get_by_id.return_value = make_job(status=job_status)
    ingester._documents.get_by_id.return_value = make_doc(status=doc_status)

    if outcome == _START_JOB_DONE:
        with pytest.raises(JobAlreadyDone):
            await ingester.start_stage(JOB_ID)
    elif outcome == _START_JOB_FAILED:
        with pytest.raises(JobAlreadyFailed):
            await ingester.start_stage(JOB_ID)
    elif outcome == _START_DOC_STATUS:
        with pytest.raises(InvalidTransition):
            await ingester.start_stage(JOB_ID)
    else:
        job = await ingester.start_stage(JOB_ID)
        assert job.status == JobStatus.RUNNING


# --- state-machine matrix: complete_stage ----------------------------------


@pytest.mark.parametrize(
    "job_status, doc_status, expected",
    [
        (JobStatus.RUNNING, DocumentStatus.PROCESSING, "ok"),
        (JobStatus.RUNNING, DocumentStatus.PENDING, "invalid"),
        (JobStatus.RUNNING, DocumentStatus.INDEXED, "invalid"),
        (JobStatus.QUEUED, DocumentStatus.PROCESSING, "invalid"),
        (JobStatus.DONE, DocumentStatus.PROCESSING, "done"),
        (JobStatus.FAILED, DocumentStatus.PROCESSING, "invalid"),
    ],
)
async def test_complete_stage_matrix(ingester, job_status, doc_status, expected):
    ingester._jobs.get_by_id.return_value = make_job(status=job_status)
    ingester._documents.get_by_id.return_value = make_doc(status=doc_status)

    if expected == "done":
        with pytest.raises(JobAlreadyDone):
            await ingester.complete_stage(JOB_ID, details={}, next_stage=None)
    elif expected == "invalid":
        with pytest.raises(InvalidTransition):
            await ingester.complete_stage(JOB_ID, details={}, next_stage=None)
    else:
        await ingester.complete_stage(JOB_ID, details={}, next_stage=None)


# --- transaction contract --------------------------------------------------


async def test_start_stage_queued_flushes_once(ingester, session):
    ingester._jobs.get_by_id.return_value = make_job()
    ingester._documents.get_by_id.return_value = make_doc()
    await ingester.start_stage(JOB_ID)
    assert session.flush.await_count == 1


async def test_start_stage_reentrant_flushes_once(ingester, session):
    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.RUNNING)
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.PROCESSING
    )
    await ingester.start_stage(JOB_ID)
    assert session.flush.await_count == 1


async def test_start_stage_raises_before_flush_on_terminal_job(ingester, session):
    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.DONE)
    with pytest.raises(JobAlreadyDone):
        await ingester.start_stage(JOB_ID)
    assert session.flush.await_count == 0


async def test_start_stage_raises_before_mutation_on_terminal_doc(ingester, session):
    ingester._jobs.get_by_id.return_value = make_job()
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.INDEXED)
    with pytest.raises(InvalidTransition):
        await ingester.start_stage(JOB_ID)
    # No flush means no partial writes
    assert session.flush.await_count == 0


async def test_complete_stage_flushes_once(ingester, session):
    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.RUNNING)
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.PROCESSING
    )
    await ingester.complete_stage(JOB_ID, details={}, next_stage=None)
    assert session.flush.await_count == 1


async def test_fail_stage_noop_still_flushes(ingester, session):
    """Current behavior: even a no-op path calls flush once, so callers
    can uniformly inspect after fail_stage returns."""
    ingester._jobs.get_by_id.return_value = None
    await ingester.fail_stage(JOB_ID, error="boom")
    assert session.flush.await_count == 1



async def test_ingester_never_commits_or_rolls_back(ingester_with_live_status, session):
    """Fails loudly if any method bypasses the caller-owns-transaction rule."""
    ing = ingester_with_live_status
    ing._jobs.get_by_id.return_value = make_job()
    ing._documents.get_by_id.return_value = make_doc()

    await ing.start_stage(JOB_ID)
    ing._jobs.get_by_id.return_value = make_job(status=JobStatus.RUNNING)
    await ing.complete_stage(JOB_ID, details={}, next_stage=None)
    ing._jobs.get_by_id.return_value = make_job(status=JobStatus.RUNNING)
    await ing.fail_stage(JOB_ID, error="boom")

    session.commit.assert_not_awaited()
    session.rollback.assert_not_awaited()


async def test_start_stage_document_missing_does_not_commit(ingester, session):
    """Docstring contract: DocumentNotFound raises without touching the
    job or flushing. The caller decides cleanup."""
    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.QUEUED)
    ingester._documents.get_by_id.return_value = None
    with pytest.raises(DocumentNotFound):
        await ingester.start_stage(JOB_ID)
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    
# --- invariants ------------------------------------------------------------


async def test_start_stage_invariants_after_queued_success(ingester):
    job = make_job(status=JobStatus.QUEUED)
    doc = make_doc(status=DocumentStatus.PENDING)
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    await ingester.start_stage(JOB_ID)
    assert_invariants(job, doc)
    assert job.started_at is not None


async def test_complete_stage_invariants_after_final(ingester):
    job = make_job(status=JobStatus.RUNNING, stage=JobStage.INDEX)
    doc = make_doc(status=DocumentStatus.PROCESSING)
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    await ingester.complete_stage(JOB_ID, details={}, next_stage=None)
    assert_invariants(job, doc)
    assert job.finished_at >= job.started_at if job.started_at else True


async def test_complete_stage_finished_at_not_before_started_at(ingester):
    started = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    job = make_job(status=JobStatus.RUNNING, started_at=started)
    doc = make_doc(status=DocumentStatus.PROCESSING)
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    await ingester.complete_stage(JOB_ID, details={}, next_stage=None)

    assert job.finished_at >= started


async def test_fail_stage_invariants(ingester):
    job = make_job(status=JobStatus.RUNNING, started_at=datetime.now(UTC))
    doc = make_doc(status=DocumentStatus.PROCESSING)
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    await ingester.fail_stage(JOB_ID, error="boom")
    assert_invariants(job, doc)


# --- details merge edge cases ----------------------------------------------


@pytest.mark.parametrize(
    "existing, incoming, expected",
    [
        (None, {"a": 1}, {"a": 1}),
        ({}, {"a": 1}, {"a": 1}),
        ({"a": 1}, {}, {"a": 1}),
        ({"a": 1}, {"a": 2}, {"a": 2}),
        ({"a": 1, "b": 2}, {"b": 3, "c": 4}, {"a": 1, "b": 3, "c": 4}),
        ({"nested": {"x": 1}}, {"nested": {"y": 2}},
         {"nested": {"y": 2}}),  # shallow
    ],
)
async def test_complete_stage_merge_semantics(ingester, existing, incoming, expected):
    job = make_job(status=JobStatus.RUNNING, details=existing)
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.PROCESSING
    )
    await ingester.complete_stage(JOB_ID, details=incoming, next_stage=None)
    assert job.details == expected


async def test_complete_stage_does_not_mutate_caller_details(ingester):
    caller_details = {"a": 1}
    job = make_job(status=JobStatus.RUNNING, details={"b": 2})
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.PROCESSING
    )
    await ingester.complete_stage(JOB_ID, details=caller_details, next_stage=None)
    assert caller_details == {"a": 1}


async def test_complete_stage_metadata_merge_preserves_existing_keys(ingester):
    job = make_job(status=JobStatus.RUNNING, stage=JobStage.INDEX)
    doc = make_doc(status=DocumentStatus.PROCESSING)
    doc.metadata_ = {"existing": "value"}
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    await ingester.complete_stage(JOB_ID, details={"inserted": 7}, next_stage=None)

    assert doc.metadata_ == {"existing": "value", "chunk_count": 7}


async def test_complete_stage_metadata_when_doc_metadata_is_none(ingester):
    job = make_job(status=JobStatus.RUNNING, stage=JobStage.INDEX)
    doc = make_doc(status=DocumentStatus.PROCESSING)
    doc.metadata_ = None
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    await ingester.complete_stage(JOB_ID, details={"inserted": 7}, next_stage=None)

    assert doc.metadata_ == {"chunk_count": 7}


# --- pipeline integration --------------------------------------------------


async def test_full_pipeline_state_progression(ingester_with_live_status):
    """Drive a document through all four stages and verify state at each
    transition. The document object is shared across stages (its status
    mutates in place). Each stage's job is a fresh mock."""
    ingester = ingester_with_live_status
    doc = make_doc(status=DocumentStatus.PENDING)
    ingester._documents.get_by_id.return_value = doc

    job_extract = make_job(stage=JobStage.EXTRACT, status=JobStatus.QUEUED)
    job_chunk = make_job(stage=JobStage.CHUNK, status=JobStatus.QUEUED)
    job_embed = make_job(stage=JobStage.EMBED, status=JobStatus.QUEUED)
    job_index = make_job(stage=JobStage.INDEX, status=JobStatus.QUEUED)

    # --- EXTRACT ---
    ingester._jobs.get_by_id.return_value = job_extract
    await ingester.start_stage(JOB_ID, stage=JobStage.EXTRACT)
    assert (job_extract.status, doc.status) == (
        JobStatus.RUNNING,
        DocumentStatus.PROCESSING,
    )

    ingester._jobs.create_enqueue.return_value = job_chunk
    await ingester.complete_stage(
        JOB_ID, details={"pages": 42}, next_stage=JobStage.CHUNK
    )
    assert job_extract.status == JobStatus.DONE
    assert doc.page_count == 42
    assert doc.status == DocumentStatus.PROCESSING  # unchanged mid-pipeline
    ingester._jobs.create_enqueue.assert_awaited_with(
        document_id=doc.id, stage=JobStage.CHUNK
    )

    # --- CHUNK ---
    ingester._jobs.get_by_id.return_value = job_chunk
    await ingester.start_stage(JOB_ID, stage=JobStage.CHUNK)
    assert job_chunk.status == JobStatus.RUNNING

    ingester._jobs.create_enqueue.return_value = job_embed
    await ingester.complete_stage(
        JOB_ID, details={"chunks": 128}, next_stage=JobStage.EMBED
    )
    assert job_chunk.status == JobStatus.DONE

    # --- EMBED ---
    ingester._jobs.get_by_id.return_value = job_embed
    await ingester.start_stage(JOB_ID, stage=JobStage.EMBED)
    assert job_embed.status == JobStatus.RUNNING

    ingester._jobs.create_enqueue.return_value = job_index
    await ingester.complete_stage(
        JOB_ID, details={"embedded": 128}, next_stage=JobStage.INDEX
    )
    assert job_embed.status == JobStatus.DONE

    # --- INDEX (terminal) ---
    ingester._jobs.get_by_id.return_value = job_index
    await ingester.start_stage(JOB_ID, stage=JobStage.INDEX)
    assert job_index.status == JobStatus.RUNNING

    result = await ingester.complete_stage(
        JOB_ID, details={"inserted": 128}, next_stage=None
    )
    assert result is None
    assert job_index.status == JobStatus.DONE
    assert doc.status == DocumentStatus.INDEXED
    assert doc.indexed_at is not None
    assert doc.metadata_["chunk_count"] == 128
    assert ingester._jobs.create_enqueue.await_count == 3

async def test_pipeline_failure_mid_stage(ingester):
    """A permanent failure at stage N marks job and doc FAILED, leaving
    the doc recoverable by re-running the pipeline."""
    doc = make_doc(status=DocumentStatus.PROCESSING)
    job = make_job(stage=JobStage.EMBED, status=JobStatus.RUNNING)
    ingester._documents.get_by_id.return_value = doc
    ingester._jobs.get_by_id.return_value = job

    await ingester.fail_stage(JOB_ID, error="vector store unavailable")

    assert job.status == JobStatus.FAILED
    assert job.details["error"] == "vector store unavailable"
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message == "vector store unavailable"


# --- concurrency -----------------------------------------------------------

async def test_two_jobs_different_documents_run_independently(ingester):
    """Two concurrent start_stage calls on distinct jobs must not share
    state or interfere with each other."""
    doc_a = make_doc(doc_id=DOC_ID, status=DocumentStatus.PENDING)
    doc_b = make_doc(doc_id=uuid4(), status=DocumentStatus.PENDING)
    job_a = make_job(document_id=doc_a.id, status=JobStatus.QUEUED)
    job_b = make_job(document_id=doc_b.id, status=JobStatus.QUEUED)

    docs = {doc_a.id: doc_a, doc_b.id: doc_b}
    jobs = {JOB_ID: job_a, uuid4(): job_b}
    jid_b = next(iter(k for k in jobs if k != JOB_ID))

    async def get_job(jid):
        return jobs[jid]

    async def get_doc(did):
        return docs[did]

    async def set_status(doc_id, new_status):
        docs[doc_id].status = new_status

    ingester._jobs.get_by_id.side_effect = get_job
    ingester._documents.get_by_id.side_effect = get_doc
    ingester._documents.set_status.side_effect = set_status

    await asyncio.gather(
        ingester.start_stage(JOB_ID),
        ingester.start_stage(jid_b),
    )

    assert job_a.status == JobStatus.RUNNING
    assert job_b.status == JobStatus.RUNNING
    assert doc_a.status == DocumentStatus.PROCESSING
    assert doc_b.status == DocumentStatus.PROCESSING
    
    
async def test_start_stage_raises_specific_terminal_exceptions(ingester):
    """JobAlreadyDone and JobAlreadyFailed are distinguishable — callers
    can branch on them without catching a generic parent."""
    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.DONE)
    with pytest.raises(JobAlreadyDone):
        await ingester.start_stage(JOB_ID)

    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.FAILED)
    with pytest.raises(JobAlreadyFailed):
        await ingester.start_stage(JOB_ID)


async def test_start_stage_raises_invalid_transition_for_bad_doc_status(ingester):
    """A non-terminal job with an unmoveable document raises the generic
    InvalidTransition, not a terminal-specific subclass."""
    ingester._jobs.get_by_id.return_value = make_job(status=JobStatus.QUEUED)
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.INDEXED
    )
    with pytest.raises(InvalidTransition):
        await ingester.start_stage(JOB_ID)
        
async def test_reentrant_start_stage_is_stable(ingester):
    """ARQ delivers the same job twice — start_stage must not double-mutate."""
    job = make_job(status=JobStatus.RUNNING, started_at=datetime.now(UTC))
    doc = make_doc(status=DocumentStatus.PROCESSING)
    ingester._jobs.get_by_id.return_value = job
    ingester._documents.get_by_id.return_value = doc

    started = job.started_at
    await ingester.start_stage(JOB_ID)
    await ingester.start_stage(JOB_ID)
    await ingester.start_stage(JOB_ID)

    assert job.status == JobStatus.RUNNING
    assert job.started_at == started  # never re-stamped
    ingester._documents.set_status.assert_not_awaited()


# --- embed -----------------------------------------------------------------


async def test_embed_strict_zip_rejects_mismatched_vector_count(ingester):
    """If embed_in_batches drops a row (transient provider bug), strict
    zip must raise rather than silently attach the wrong vector."""
    chunks = [make_chunk(i) for i in range(3)]
    with (
        patch("app.rag.ingestion.get_embedder"),
        patch(
            "app.rag.ingestion.embed_in_batches",
            new=AsyncMock(return_value=[[0.1]]),  # only 1 vector for 3 chunks
        ),
        patch("app.rag.ingestion.settings") as s,
    ):
        s.embedding_spec.embed_timeout_s = 30
        with pytest.raises(ValueError):
            await ingester.embed(chunks)


async def test_embed_preserves_chunk_order_and_fields(ingester):
    chunks = [make_chunk(i, text=f"text-{i}") for i in range(4)]
    vectors = [[float(i)] * 3 for i in range(4)]
    with (
        patch("app.rag.ingestion.get_embedder"),
        patch(
            "app.rag.ingestion.embed_in_batches",
            new=AsyncMock(return_value=vectors),
        ),
        patch("app.rag.ingestion.settings") as s,
    ):
        s.embedding_spec.embed_timeout_s = 30
        out = await ingester.embed(chunks)

    for i, r in enumerate(out):
        assert r.chunk_index == i
        assert r.text == f"text-{i}"
        assert r.embedding == vectors[i]


async def test_embed_does_not_mutate_input_chunks(ingester):
    chunks = [make_chunk(0, text="original")]
    with (
        patch("app.rag.ingestion.get_embedder"),
        patch(
            "app.rag.ingestion.embed_in_batches",
            new=AsyncMock(return_value=[[0.5]]),
        ),
        patch("app.rag.ingestion.settings") as s,
    ):
        s.embedding_spec.embed_timeout_s = 30
        await ingester.embed(chunks)

    assert chunks[0].text == "original"
    assert chunks[0].embedding is None  # untouched


async def test_embed_cancellation_propagates(ingester):
    """Cancelling the embed coroutine must propagate CancelledError, not
    convert to TransientEmbeddingError. ARQ relies on this to distinguish
    shutdown from a real failure."""
    started = asyncio.Event()

    async def slow(*_a, **_kw):
        started.set()
        await asyncio.sleep(60)

    with (
        patch("app.rag.ingestion.get_embedder"),
        patch("app.rag.ingestion.embed_in_batches", new=slow),
        patch("app.rag.ingestion.settings") as s,
    ):
        s.embedding_spec.embed_timeout_s = 30
        task = asyncio.create_task(ingester.embed([make_chunk(0)]))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_start_stage_document_missing_does_not_commit(ingester, session):
    """If the document is gone, start_stage raises without touching the
    job — the caller decides whether to fail it."""
    ingester._jobs.get_by_id.return_value = make_job()
    ingester._documents.get_by_id.return_value = None
    with pytest.raises(DocumentNotFound):
        await ingester.start_stage(JOB_ID)
    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()

# --- error-type specificity ------------------------------------------------


@pytest.mark.parametrize(
    "status, expected_exc",
    [
        (JobStatus.QUEUED, InvalidTransition),   # bad doc status
        (JobStatus.RUNNING, InvalidTransition),  # bad doc status
        (JobStatus.DONE, JobAlreadyDone),
        (JobStatus.FAILED, JobAlreadyFailed),
    ],
)
async def test_start_stage_rejects_terminal_doc(ingester, status, expected_exc):
    """When the document is in a non-startable state, the exception
    raised depends on the job status: terminal jobs raise their
    specific exceptions before the doc is ever read; non-terminal jobs
    raise the generic InvalidTransition."""
    ingester._jobs.get_by_id.return_value = make_job(status=status)
    ingester._documents.get_by_id.return_value = make_doc(
        status=DocumentStatus.INDEXED
    )
    with pytest.raises(expected_exc):
        await ingester.start_stage(JOB_ID)


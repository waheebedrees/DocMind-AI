from pydantic import BaseModel, model_validator


class RetrievalSettings(BaseModel):
    """Knobs for the retrieval pipeline.

    All values are defaults; a caller wanting different behavior for a
    single request should construct a ``RetrievalService`` and override
    ``service.cfg`` before calling ``retrieve``. There is no per-request
    settings parameter by design — tuning is a deployment concern.

    Attributes:
        vector_candidates: How many vector-search hits to fetch before
            fusion. Higher = better recall, more DB work.
        keyword_candidates: Same, for the keyword search.
        fts_language: Postgres text-search configuration. Must match the
            language the ``text_search`` column was populated with.
        hnsw_ef_search: pgvector's ``hnsw.ef_search`` parameter. Higher
            = better recall, slower.
        hnsw_iterative_scan: If True, sets ``hnsw.iterative_scan`` to
            ``relaxed_order`` so the vector search keeps scanning until
            it has enough results *after* the user filter. Requires
            pgvector >= 0.8. Without it, tenants with few documents get
            fewer than ``vector_candidates`` results.
        rrf_k: Reciprocal Rank Fusion smoothing constant. 60 is the
            standard value; lower weights top ranks more heavily.
        vector_weight: RRF weight for the vector list.
        keyword_weight: RRF weight for the keyword list.
        rerank_enabled: Master switch for the rerank stage.
        rerank_url: Endpoint of the reranker (e.g. TEI). Not used when
            ``rerank_enabled`` is False.
        rerank_candidates: How many fused candidates to send to the
            reranker. Capped at this to bound rerank latency.
        rerank_timeout_s: Per-request timeout for the reranker call.
        rerank_min_score: If set, candidates scoring below this after
            rerank are dropped. ``None`` keeps all reranked candidates.
            Calibrate on your eval set; the cross-encoder's score scale
            is model-specific.
        mmr_lambda: MMR trade-off. 1.0 = pure relevance, 0.0 = pure
            diversity. 0.7 is a reasonable default for prose.
        neighbor_window: How many chunks on either side of a selected
            chunk to fetch for context expansion. 0 disables expansion.
        max_context_tokens: Soft cap on the rendered context length.
            ``build_passages`` trims to fit.
    """

    # candidate generation
    vector_candidates: int = 50
    keyword_candidates: int = 50
    fts_language: str = "english"
    hnsw_ef_search: int = 100
    hnsw_iterative_scan: bool = True  # requires pgvector >= 0.8

    # fusion
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0

    # rerank
    rerank_enabled: bool = False

    # e.g. TEI: http://reranker:8080/rerank
    rerank_url: str | None = None
    rerank_candidates: int = 30
    rerank_timeout_s: float = 20.0
    # model-specific; calibrate on your eval set
    rerank_min_score: float | None = None

    # selection
    mmr_lambda: float = 0.7
    neighbor_window: int = 1
    max_context_tokens: int = 6000

    @model_validator(mode="after")
    def _validate_cross_field(self) -> "RetrievalSettings":
        if self.vector_weight == 0 and self.keyword_weight == 0:
            raise ValueError("vector_weight and keyword_weight cannot both be zero — fusion would produce no candidates")
        if self.rerank_enabled and self.rerank_url is None:
            raise ValueError("rerank_enabled=True requires rerank_url to be set")
        if self.rerank_min_score is not None and not self.rerank_enabled:
            raise ValueError("rerank_min_score is set but rerank_enabled=False — the threshold will never apply")
        return self

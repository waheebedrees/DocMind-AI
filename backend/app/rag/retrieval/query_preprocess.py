"""Query-side preprocessing.

Centralizes everything that happens to a raw user string before it
reaches the database: normalization, empty-checking, and tsquery
construction. Kept separate from ``RetrievalService`` so it can be
unit-tested without a DB session, and so the same preprocessing is
guaranteed to apply to every search path.
"""

from __future__ import annotations

import re

from sqlalchemy import Text, cast, func
from sqlalchemy.dialects.postgresql import TSQUERY


class QueryPreprocess:
    """Normalize a raw query and build the tsquery for keyword search.

    One instance per process is enough — the class holds no mutable
    state. The ``language`` is bound at construction so callers can't
    accidentally use a different text-search config for indexing vs.
    querying.

    Usage::

        qp = QueryPreprocess(language="english")

        normalized = qp.normalize(raw_query)   # raises on empty
        tsq = qp.to_tsquery(normalized)        # None if no word chars
        if tsq is None:
            return []

    Attributes:
        language: Postgres text-search configuration name. Must match
            the one used when populating ``DocumentChunk.text_search``.
    """

    # Any unicode word character. Used to short-circuit pure-punctuation
    # or whitespace-only queries, which ``plainto_tsquery`` would turn
    # into an empty tsquery — valid but wasteful to send to Postgres.
    _WORD_RE = re.compile(r"\w", re.UNICODE)

    # Websearch operator syntax: quoted phrases, negation with '-', or
    # the bare word OR. Used only by ``has_operator_syntax``; the
    # default path does NOT branch on this — see module docstring.
    _OPERATOR_RE = re.compile(r'"[^"]*"|(?:^|\s)-\w+|\bOR\b')

    def __init__(self, language: str = "english") -> None:
        self.language = language

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, query: str) -> str:
        """Collapse whitespace and reject empty queries.

        Runs on every query before anything else. Whitespace collapse
        matters because FTS treats newlines and tabs oddly in some
        edge cases, and because a query of ``"   "`` should be a
        client error, not a full table scan.

        Args:
            query: Raw user input. May contain leading/trailing
                whitespace, internal newlines, tabs, or non-breaking
                spaces.

        Returns:
            The query with all runs of whitespace collapsed to single
            ASCII spaces, stripped at the ends.

        Raises:
            ValueError: The query is empty after normalization.
        """
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    # ------------------------------------------------------------------
    # tsquery construction
    # ------------------------------------------------------------------

    def to_tsquery(self, query: str):
        """Build an OR tsquery using Postgres's own parser.

        ``plainto_tsquery`` ANDs every lexeme, which almost never
        matches a single chunk for a natural-language question: one
        stemmed word absent from the chunk kills the whole match.
        Rewriting ``&`` to ``|`` keeps Postgres's stemming, stop-word
        removal, and unicode handling while letting a partial match
        surface; ``ts_rank_cd`` then orders by match density.

        The cast chain is intentional:

        * ``plainto_tsquery(...)`` returns ``tsquery``.
        * ``cast(..., Text)`` turns it into the serialized string form
          (e.g. ``'cpu & requir & zerostrik'``).
        * ``replace(..., '&', '|')`` swaps the operator.
        * ``cast(..., TSQUERY)`` parses it back into a tsquery.

        Args:
            query: A normalized query string (see ``normalize``).
                Does not need to be re-normalized by the caller.

        Returns:
            A SQLAlchemy expression producing a ``tsquery`` with OR
            semantics, or ``None`` if the query contains no word
            characters at all (pure punctuation/whitespace).
        """
        if not self._WORD_RE.search(query):
            return None
        return cast(
            func.replace(
                cast(func.plainto_tsquery(self.language, query), Text),
                "&",
                "|",
            ),
            TSQUERY,
        )

    # ------------------------------------------------------------------
    # Optional: websearch-operator detection
    # ------------------------------------------------------------------

    def has_operator_syntax(self, query: str) -> bool:
        """Return True if the query looks like it uses websearch syntax.

        Detects quoted phrases, ``-exclusion``, or bare ``OR``. Not
        used by ``to_tsquery`` — the default pipeline always uses OR
        semantics. Provided for callers that want to offer an
        "advanced search" mode backed by ``websearch_to_tsquery``.

        Warning: the bare-word ``OR`` check fires on any sentence
        containing the English word "or", e.g. "either X or Y". Only
        use this if you gate it behind an explicit UI toggle; do not
        auto-branch on it in a general Q&A pipeline.

        Args:
            query: A normalized query string.

        Returns:
            True if any operator pattern matches, else False.
        """
        return bool(self._OPERATOR_RE.search(query))

    def to_websearch_tsquery(self, query: str):
        """Build a tsquery from websearch syntax.

        Use only when the caller has confirmed via
        ``has_operator_syntax`` that the user intends operator
        semantics. ``websearch_to_tsquery`` never raises on malformed
        input — it degrades to treating unrecognized tokens as
        plain terms.

        Args:
            query: A normalized query string.

        Returns:
            A SQLAlchemy expression producing a ``tsquery``, or
            ``None`` if the query contains no word characters.
        """
        if not self._WORD_RE.search(query):
            return None
        return func.websearch_to_tsquery(self.language, query)

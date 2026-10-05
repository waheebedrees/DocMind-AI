from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.models.message import Message


class Citation(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "citations"

    message_id: Mapped[UUID] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True, nullable=False)
    chunk_id: Mapped[UUID] = mapped_column(ForeignKey("document_chunks.id", ondelete="CASCADE"), index=True, nullable=False)

    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    message: Mapped["Message"] = relationship(back_populates="citations")

    __table_args__ = (
        UniqueConstraint("message_id", "chunk_id", name="uq_citation_message_chunk"),
        CheckConstraint("rank >= 1", name="rank_positive"),
        CheckConstraint("score >= 0 AND score <= 1", name="score_range"),
        CheckConstraint("rank <= 100", name="rank_max"),
    )

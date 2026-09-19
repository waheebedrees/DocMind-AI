from uuid import UUID

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class EvaluationRun(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_runs"

    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class EvaluationResult(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_results"

    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

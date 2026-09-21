from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column


def utc_now():
    return datetime.now(UTC)


class UUIDMixin:
    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=func.gen_random_uuid(),
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.clock_timestamp(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(
        timezone=True), server_default=func.clock_timestamp(), onupdate=func.now(), nullable=False)

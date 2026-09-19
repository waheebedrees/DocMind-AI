from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class ProcessingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"

    version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

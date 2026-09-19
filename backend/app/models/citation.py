from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class Citation(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "citations"

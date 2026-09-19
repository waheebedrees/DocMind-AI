from app.db.base import Base
from app.models.mixins import TimestampMixin, UUIDMixin


class Extraction(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "extractions"

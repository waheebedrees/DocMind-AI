from enum import StrEnum

from sqlalchemy import Enum as SAEnum


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class JobStage(StrEnum):
    EXTRACT = "extract"
    CLEAN = "clean"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class EvaluationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def enum_column(enum_cls: type[StrEnum], name: str) -> SAEnum:
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        values_callable=lambda e: [m.value for m in e],
        length=32,
    )


_STAGE_ORDER: tuple[JobStage, ...] = (
    JobStage.EXTRACT,
    JobStage.CLEAN,
    JobStage.CHUNK,
    JobStage.EMBED,
    JobStage.INDEX,
)

_STAGE_SEQUENCE: tuple[tuple[JobStage, str], ...] = (
    (JobStage.EXTRACT, "process_extract"),
    (JobStage.CLEAN, "process_clean"),
    (JobStage.CHUNK, "process_chunk"),
    (JobStage.EMBED, "process_embed"),
    (JobStage.INDEX, "process_index"),
)
_STAGE_INDEX = {stage: i for i, (stage, _) in enumerate(_STAGE_SEQUENCE)}
_STAGE_TASK = dict(_STAGE_SEQUENCE)


_STAGE_ORDER_VALUES = tuple(s.value for s in _STAGE_ORDER)

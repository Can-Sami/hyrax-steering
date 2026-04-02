from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Intent(Base):
    __tablename__ = 'intents'

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    intent_code: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IntentUtterance(Base):
    __tablename__ = 'intent_utterances'

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    intent_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('intents.id', ondelete='CASCADE'),
        nullable=False,
    )
    language_code: Mapped[str] = mapped_column(String(8), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IntentEmbedding(Base):
    __tablename__ = 'intent_embeddings'

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    utterance_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('intent_utterances.id', ondelete='CASCADE'),
        nullable=False,
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    norm: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint('utterance_id', 'model_name', name='uq_intent_embeddings_utterance_model'),
    )


class InferenceRequest(Base):
    __tablename__ = 'inference_requests'

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    external_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    language_code: Mapped[str] = mapped_column(String(8), nullable=False)
    audio_uri: Mapped[str] = mapped_column(Text, nullable=False)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    processing_ms: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint('external_request_id', name='uq_inference_requests_external_request_id'),
    )


class InferenceResult(Base):
    __tablename__ = 'inference_results'

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('inference_requests.id', ondelete='CASCADE'),
        nullable=False,
    )
    predicted_intent_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey('intents.id', ondelete='SET NULL'),
        nullable=True,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_k_json: Mapped[dict | list] = mapped_column(JSONB, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint('request_id', name='uq_inference_results_request_id'),
    )


class ModelRegistry(Base):
    __tablename__ = 'model_registry'

    id: Mapped[str] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    model_key: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    device_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    quantization: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint('model_key', 'version', 'device_mode', name='uq_model_registry_key_version_device'),
    )

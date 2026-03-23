"""SQLAlchemy ORM models for TempusSensor.

Cloud-ready: works with PostgreSQL (Render) and SQLite (local).
Knowledge graph stored in relational tables (replaces Neo4j).
"""

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db import Base


class PaperType(str, enum.Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    TEORICO = "TEORICO"
    REVIEW = "REVIEW"
    SIMULACION = "SIMULACION"


class HypothesisType(str, enum.Enum):
    INCREMENTAL = "INCREMENTAL"
    COMBINATORIA = "COMBINATORIA"
    RADICAL = "RADICAL"


class HypothesisStatus(str, enum.Enum):
    PENDING = "PENDING"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    REFUTED = "REFUTED"


class EvidenceClass(str, enum.Enum):
    DATO_EXPERIMENTAL = "DATO_EXPERIMENTAL"
    RESULTADO_SIMULACION = "RESULTADO_SIMULACION"
    INFERENCIA = "INFERENCIA"
    EXTRAPOLACION = "EXTRAPOLACION"
    SIN_SOPORTE = "SIN_SOPORTE"


class IntegrityCertification(str, enum.Enum):
    INTEGRO = "INTEGRO"
    CON_ADVERTENCIA = "CON_ADVERTENCIA"
    NO_CONFIABLE = "NO_CONFIABLE"


class MaterialType(str, enum.Enum):
    NV_DIAMOND = "NV_DIAMOND"
    TRAPPED_ION = "TRAPPED_ION"
    SUPERCONDUCTOR = "SUPERCONDUCTOR"
    OTHER = "OTHER"


class PredictionStatus(str, enum.Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    EXPIRED = "EXPIRED"


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doi: Mapped[str | None] = mapped_column(String(256), unique=True, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    titulo: Mapped[str] = mapped_column(Text, nullable=False)
    autores: Mapped[str] = mapped_column(Text, nullable=False)
    institucion: Mapped[str | None] = mapped_column(Text, nullable=True)
    fecha: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    journal: Mapped[str | None] = mapped_column(String(256), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)

    tipo: Mapped[str] = mapped_column(String(32), nullable=False)

    parametros_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    reproducibilidad: Mapped[float] = mapped_column(Float, default=0.0)
    novedad: Mapped[float] = mapped_column(Float, default=0.0)
    relevancia_biosensado: Mapped[float] = mapped_column(Float, default=0.0)

    confianza_fuente: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    hypotheses: Mapped[list["Hypothesis"]] = relationship(back_populates="source_paper")
    simulations: Mapped[list["Simulation"]] = relationship(back_populates="paper")


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    enunciado: Mapped[str] = mapped_column(Text, nullable=False)
    accion: Mapped[str] = mapped_column(Text, nullable=False)
    resultado_esperado: Mapped[str] = mapped_column(Text, nullable=False)
    mecanismo: Mapped[str] = mapped_column(Text, nullable=False)

    tipo: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")

    confianza: Mapped[float] = mapped_column(Float, nullable=False)
    impacto: Mapped[float] = mapped_column(Float, default=0.5)
    costo_testeo: Mapped[float] = mapped_column(Float, default=0.5)
    rank_score: Mapped[float] = mapped_column(Float, default=0.0)

    soporte_teorico: Mapped[str | None] = mapped_column(Text, nullable=True)
    soporte_experimental: Mapped[str | None] = mapped_column(Text, nullable=True)

    source_paper_id: Mapped[int | None] = mapped_column(ForeignKey("papers.id"), nullable=True)
    source_paper: Mapped[Paper | None] = relationship(back_populates="hypotheses")

    evidence_class: Mapped[str] = mapped_column(String(32), default="INFERENCIA")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int | None] = mapped_column(ForeignKey("papers.id"), nullable=True)
    paper: Mapped[Paper | None] = relationship(back_populates="simulations")

    material_type: Mapped[str] = mapped_column(String(32), nullable=False)
    temperatura_k: Mapped[float] = mapped_column(Float, nullable=False)
    potencia_laser_w: Mapped[float] = mapped_column(Float, nullable=False)
    campo_externo_t: Mapped[float] = mapped_column(Float, default=0.0)
    densidad_defectos: Mapped[float] = mapped_column(Float, default=0.0)
    n_qubits: Mapped[int] = mapped_column(Integer, default=10)

    sensibilidad_05_10hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    sensibilidad_10_50hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    sensibilidad_50_100hz: Mapped[float | None] = mapped_column(Float, nullable=True)

    r_squared: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)

    confianza: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_class: Mapped[str] = mapped_column(String(32), default="RESULTADO_SIMULACION")

    pareto_optimal: Mapped[bool] = mapped_column(Boolean, default=False)
    parameters_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Protocol(Base):
    __tablename__ = "protocols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    titulo: Mapped[str] = mapped_column(Text, nullable=False)
    objetivo: Mapped[str] = mapped_column(Text, nullable=False)

    material: Mapped[str] = mapped_column(Text, nullable=False)
    material_type: Mapped[str] = mapped_column(String(32), nullable=False)
    sensor_config: Mapped[str] = mapped_column(Text, nullable=False)
    driving_config: Mapped[str] = mapped_column(Text, nullable=False)
    detection_method: Mapped[str] = mapped_column(Text, nullable=False)

    pasos: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metricas_exito: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    seguridad: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    sensibilidad_predicha_pt: Mapped[float | None] = mapped_column(Float, nullable=True)
    temperatura_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    costo_estimado_usd: Mapped[float | None] = mapped_column(Float, nullable=True)

    confianza: Mapped[float] = mapped_column(Float, default=0.5)
    evidence_class: Mapped[str] = mapped_column(String(32), default="INFERENCIA")

    simulation_id: Mapped[int | None] = mapped_column(ForeignKey("simulations.id"), nullable=True)
    hypothesis_id: Mapped[int | None] = mapped_column(ForeignKey("hypotheses.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


class IntegrityCheck(Base):
    __tablename__ = "integrity_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    fidelidad_dtc: Mapped[float] = mapped_column(Float, nullable=False)
    coherencia_umbral: Mapped[float] = mapped_column(Float, default=0.8)
    coherencia_estado: Mapped[str] = mapped_column(String(16), nullable=False)

    correlacion_media: Mapped[float] = mapped_column(Float, nullable=False)
    correlacion_umbral: Mapped[float] = mapped_column(Float, default=0.6)
    correlacion_estado: Mapped[str] = mapped_column(String(16), nullable=False)

    divergencia_kl: Mapped[float] = mapped_column(Float, nullable=False)
    ruido_umbral: Mapped[float] = mapped_column(Float, default=0.5)
    ruido_estado: Mapped[str] = mapped_column(String(16), nullable=False)
    fuentes_interferencia: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    certificacion: Mapped[str] = mapped_column(String(32), nullable=False)
    hash_datos: Mapped[str] = mapped_column(String(128), nullable=False)
    version_protocolo: Mapped[str] = mapped_column(String(32), default="1.0")

    simulation_id: Mapped[int | None] = mapped_column(ForeignKey("simulations.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Agent cycle log
# ---------------------------------------------------------------------------


class AgentCycle(Base):
    __tablename__ = "agent_cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    lesson_learned: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Prediction tracking
# ---------------------------------------------------------------------------


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediccion: Mapped[str] = mapped_column(Text, nullable=False)
    probabilidad: Mapped[float] = mapped_column(Float, nullable=False)
    horizonte_dias: Mapped[int] = mapped_column(Integer, default=30)
    base_evidencia: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    resultado_real: Mapped[str | None] = mapped_column(Text, nullable=True)
    hypothesis_id: Mapped[int | None] = mapped_column(ForeignKey("hypotheses.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Knowledge Graph in PostgreSQL (replaces Neo4j)
# ---------------------------------------------------------------------------


class KnowledgeNode(Base):
    """Generic knowledge graph node — Materials, Techniques, Applications, etc."""
    __tablename__ = "knowledge_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # MATERIAL | TECHNIQUE | APPLICATION | RESULT | PREDICTION
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    properties: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class KnowledgeEdge(Base):
    """Edge between knowledge nodes."""
    __tablename__ = "knowledge_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("knowledge_nodes.id"), nullable=False)
    target_id: Mapped[int] = mapped_column(ForeignKey("knowledge_nodes.id"), nullable=False)
    relation: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # LOGRA, PRODUCE, REPORTA, VALIDA, REFUTA, HABILITA, COMPITE_CON, BLOQUEA, REQUIERE
    properties: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

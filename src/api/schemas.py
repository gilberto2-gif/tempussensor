"""Pydantic schemas — EVERY model carries mandatory confidence fields."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PaperType(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    TEORICO = "TEORICO"
    REVIEW = "REVIEW"
    SIMULACION = "SIMULACION"


class HypothesisType(str, Enum):
    INCREMENTAL = "INCREMENTAL"
    COMBINATORIA = "COMBINATORIA"
    RADICAL = "RADICAL"


class HypothesisStatus(str, Enum):
    PENDING = "PENDING"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    REFUTED = "REFUTED"


class EvidenceClass(str, Enum):
    DATO_EXPERIMENTAL = "DATO_EXPERIMENTAL"
    RESULTADO_SIMULACION = "RESULTADO_SIMULACION"
    INFERENCIA = "INFERENCIA"
    EXTRAPOLACION = "EXTRAPOLACION"
    SIN_SOPORTE = "SIN_SOPORTE"


class IntegrityCertification(str, Enum):
    INTEGRO = "INTEGRO"
    CON_ADVERTENCIA = "CON_ADVERTENCIA"
    NO_CONFIABLE = "NO_CONFIABLE"


class MaterialType(str, Enum):
    NV_DIAMOND = "NV_DIAMOND"
    TRAPPED_ION = "TRAPPED_ION"
    SUPERCONDUCTOR = "SUPERCONDUCTOR"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Confidence mixin — mandatory on EVERY output schema
# ---------------------------------------------------------------------------


class ConfidenceMixin(BaseModel):
    confianza: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level 0-1. <0.7 = low confidence."
    )
    confianza_baja: bool = Field(
        default=False,
        description="Auto-set True if confianza < 0.7 — UI shows yellow badge.",
    )

    @field_validator("confianza_baja", mode="before")
    @classmethod
    def _set_low_confidence_flag(cls, v, info):
        conf = info.data.get("confianza")
        if conf is not None and conf < 0.7:
            return True
        return v if v is not None else False


# ---------------------------------------------------------------------------
# Paper schemas
# ---------------------------------------------------------------------------


class SensorParameters(BaseModel):
    material: str | None = None
    tipo_material: MaterialType | None = None
    temperatura_K: float | None = None
    sensibilidad_pT: float | None = None
    frecuencia_Hz: float | None = None
    SNR: float | None = None
    T2_us: float | None = None
    n_qubits: int | None = None
    geometria_red: str | None = None
    driving_tipo: str | None = None
    driving_frecuencia: float | None = None
    driving_potencia: float | None = None
    condiciones_ambientales: str | None = None


class PaperQuality(BaseModel):
    reproducibilidad: float = Field(0.0, ge=0.0, le=1.0)
    novedad: float = Field(0.0, ge=0.0, le=1.0)
    relevancia_biosensado: float = Field(0.0, ge=0.0, le=1.0)


class PaperCreate(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    titulo: str
    autores: str
    institucion: str | None = None
    fecha: datetime
    journal: str | None = None
    abstract: str | None = None
    tipo: PaperType
    parametros: SensorParameters | None = None
    calidad: PaperQuality = PaperQuality()
    confianza_fuente: float = Field(0.5, ge=0.0, le=1.0)


class PaperResponse(ConfidenceMixin):
    id: int
    doi: str | None = None
    arxiv_id: str | None = None
    titulo: str
    autores: str
    fecha: datetime
    tipo: PaperType
    parametros: SensorParameters | None = None
    calidad: PaperQuality
    confianza: float = Field(..., alias="confianza_fuente")

    model_config = {"from_attributes": True, "populate_by_name": True}


# ---------------------------------------------------------------------------
# Hypothesis schemas
# ---------------------------------------------------------------------------


class HypothesisCreate(BaseModel):
    accion: str
    resultado_esperado: str
    mecanismo: str
    tipo: HypothesisType
    confianza: float = Field(..., ge=0.0, le=1.0)
    impacto: float = Field(0.5, ge=0.0, le=1.0)
    costo_testeo: float = Field(0.5, ge=0.0, le=1.0)
    source_paper_id: int | None = None


class HypothesisResponse(ConfidenceMixin):
    id: int
    enunciado: str
    accion: str
    resultado_esperado: str
    mecanismo: str
    tipo: HypothesisType
    status: HypothesisStatus
    confianza: float
    impacto: float
    costo_testeo: float
    rank_score: float
    evidence_class: EvidenceClass
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Simulation schemas
# ---------------------------------------------------------------------------


class SimulationCreate(BaseModel):
    paper_id: int | None = None
    material_type: MaterialType
    temperatura_k: float
    potencia_laser_w: float
    campo_externo_t: float = 0.0
    densidad_defectos: float = 0.0
    n_qubits: int = 10


class SimulationResponse(ConfidenceMixin):
    id: int
    material_type: MaterialType
    temperatura_k: float
    potencia_laser_w: float
    sensibilidad_05_10hz: float | None = None
    sensibilidad_10_50hz: float | None = None
    sensibilidad_50_100hz: float | None = None
    r_squared: float | None = None
    rmse: float | None = None
    confianza: float
    evidence_class: EvidenceClass
    pareto_optimal: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Protocol schemas
# ---------------------------------------------------------------------------


class ProtocolResponse(ConfidenceMixin):
    id: int
    titulo: str
    objetivo: str
    material: str
    material_type: MaterialType
    sensor_config: str
    driving_config: str
    detection_method: str
    pasos: list
    metricas_exito: dict
    seguridad: list
    sensibilidad_predicha_pt: float | None = None
    temperatura_k: float | None = None
    costo_estimado_usd: float | None = None
    confianza: float
    evidence_class: EvidenceClass
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Integrity schemas
# ---------------------------------------------------------------------------


class IntegrityCheckResponse(BaseModel):
    id: int
    fidelidad_dtc: float
    coherencia_estado: str
    correlacion_media: float
    correlacion_estado: str
    divergencia_kl: float
    ruido_estado: str
    fuentes_interferencia: list | None = None
    certificacion: IntegrityCertification
    hash_datos: str
    version_protocolo: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Agent status
# ---------------------------------------------------------------------------


class AgentStatus(BaseModel):
    cycle_number: int
    phase: str
    status: str
    papers_ingested: int = 0
    hypotheses_generated: int = 0
    simulations_run: int = 0
    protocols_generated: int = 0
    prediction_accuracy: float | None = None
    last_cycle_at: datetime | None = None


# ---------------------------------------------------------------------------
# Clinical requirements (hardcoded reference)
# ---------------------------------------------------------------------------


class ClinicalRequirements(BaseModel):
    meg_sensitivity_ft: float = 10.0
    mcg_sensitivity_ft: float = 50.0
    biomarcador_sensitivity_pt: float = 1.0
    target_temp_k: float = 250.0
    target_volume_l: float = 1.0
    target_weight_kg: float = 5.0
    target_cost_usd: float = 50_000.0
    meg_installations_global: int = 200
    meg_cost_usd: float = 2_000_000.0
    epilepsy_patients_millions: float = 50.0
    alzheimer_patients_millions: float = 55.0

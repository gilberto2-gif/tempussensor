"""Confidence propagation system.

EVERY assertion in TempusSensor carries a confidence score 0-1.
Confidence < 0.7 → flagged as "low confidence" in UI with yellow badge.

Propagation rules:
- Combined confidence = product of independent confidences
- Derived assertion ≤ min(source confidences)
- Simulation results capped by model validation R²
- Extrapolations automatically penalized
"""

from enum import Enum

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)

LOW_THRESHOLD = settings.confidence_low_threshold  # 0.7


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"           # >= 0.7
    LOW = "LOW"             # 0.5 - 0.7
    VERY_LOW = "VERY_LOW"   # < 0.5


class EvidenceType(str, Enum):
    DATO_EXPERIMENTAL = "DATO_EXPERIMENTAL"
    RESULTADO_SIMULACION = "RESULTADO_SIMULACION"
    INFERENCIA = "INFERENCIA"
    EXTRAPOLACION = "EXTRAPOLACION"
    SIN_SOPORTE = "SIN_SOPORTE"


# Penalty factors for different evidence types
EVIDENCE_PENALTIES = {
    EvidenceType.DATO_EXPERIMENTAL: 1.0,       # No penalty
    EvidenceType.RESULTADO_SIMULACION: 0.85,   # 15% penalty
    EvidenceType.INFERENCIA: 0.7,              # 30% penalty
    EvidenceType.EXTRAPOLACION: 0.5,           # 50% penalty
    EvidenceType.SIN_SOPORTE: 0.0,             # Rejected
}


def classify_confidence(score: float) -> ConfidenceLevel:
    """Classify a confidence score into a level."""
    if score >= LOW_THRESHOLD:
        return ConfidenceLevel.HIGH
    elif score >= 0.5:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def is_low_confidence(score: float) -> bool:
    """Check if a score should be flagged as low confidence."""
    return score < LOW_THRESHOLD


def propagate_confidence(
    source_confidences: list[float],
    evidence_type: EvidenceType = EvidenceType.INFERENCIA,
    model_r_squared: float | None = None,
) -> float:
    """Propagate confidence through a chain of assertions.

    Rules:
    1. Base: minimum of all source confidences
    2. Apply evidence type penalty
    3. If simulation: cap by model R²
    4. Always clamp to [0, 1]
    """
    if not source_confidences:
        return 0.0

    # Rule 1: base = min of sources
    base = min(source_confidences)

    # Rule 2: evidence type penalty
    penalty = EVIDENCE_PENALTIES.get(evidence_type, 0.5)
    if penalty == 0.0:
        logger.warning("sin_soporte_rejected", source_confidences=source_confidences)
        return 0.0

    result = base * penalty

    # Rule 3: cap by model validation
    if model_r_squared is not None and evidence_type == EvidenceType.RESULTADO_SIMULACION:
        result = min(result, model_r_squared)

    # Clamp
    result = max(0.0, min(1.0, result))

    level = classify_confidence(result)
    if level != ConfidenceLevel.HIGH:
        logger.info(
            "low_confidence_propagated",
            result=round(result, 3),
            level=level.value,
            sources=source_confidences,
            evidence=evidence_type.value,
        )

    return result


def combine_independent(confidences: list[float]) -> float:
    """Combine independent confidence scores (product rule)."""
    result = 1.0
    for c in confidences:
        result *= c
    return max(0.0, min(1.0, result))


def weighted_confidence(
    scores_and_weights: list[tuple[float, float]],
) -> float:
    """Weighted average of confidence scores."""
    if not scores_and_weights:
        return 0.0
    total_weight = sum(w for _, w in scores_and_weights)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(s * w for s, w in scores_and_weights)
    return max(0.0, min(1.0, weighted_sum / total_weight))


def audit_assertion(
    claim: str,
    source_confidences: list[float],
    evidence_type: EvidenceType,
    source_papers: list[str] | None = None,
) -> dict:
    """Full audit trail for an assertion.

    Returns dict with all traceability info for Capa 6 (GOBERNAR).
    """
    conf = propagate_confidence(source_confidences, evidence_type)
    level = classify_confidence(conf)

    audit = {
        "claim": claim,
        "confidence": conf,
        "confidence_level": level.value,
        "is_low_confidence": is_low_confidence(conf),
        "evidence_type": evidence_type.value,
        "source_confidences": source_confidences,
        "source_papers": source_papers or [],
        "accepted": evidence_type != EvidenceType.SIN_SOPORTE,
    }

    if not audit["accepted"]:
        audit["rejection_reason"] = "SIN_SOPORTE assertions are rejected per governance policy."

    return audit


# Key reference confidences (hardcoded from project spec)
SOURCE_CONFIDENCE = {
    "dtc_sensor_paper": 0.52,       # Main DTC as sensor paper
    "bennett_brassard": 0.74,       # Bennett/Brassard crypto-quantum
    "microwave_quantum_network": 0.52,  # Thermal resilience in quantum networks
}

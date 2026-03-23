"""Tests for agent core: confidence system, prompts, and cycle logic."""

import pytest

from src.agent.confidence import (
    ConfidenceLevel,
    EvidenceType,
    SOURCE_CONFIDENCE,
    audit_assertion,
    classify_confidence,
    combine_independent,
    is_low_confidence,
    propagate_confidence,
    weighted_confidence,
)
from src.agent.prompts import (
    DECIDIR_PROMPT,
    EJECUTAR_PROMPT,
    GOBERNAR_PROMPT,
    MODELAR_PROMPT,
    PERCIBIR_PROMPT,
    PLANIFICAR_PROMPT,
    RECORDAR_PROMPT,
    REFLEXIONAR_PROMPT,
    SENTIR_PROMPT,
    SYSTEM_PROMPT,
)


class TestConfidence:
    def test_classify_high(self):
        assert classify_confidence(0.8) == ConfidenceLevel.HIGH
        assert classify_confidence(0.7) == ConfidenceLevel.HIGH

    def test_classify_low(self):
        assert classify_confidence(0.6) == ConfidenceLevel.LOW
        assert classify_confidence(0.5) == ConfidenceLevel.LOW

    def test_classify_very_low(self):
        assert classify_confidence(0.3) == ConfidenceLevel.VERY_LOW
        assert classify_confidence(0.0) == ConfidenceLevel.VERY_LOW

    def test_is_low_confidence(self):
        assert is_low_confidence(0.5) is True
        assert is_low_confidence(0.69) is True
        assert is_low_confidence(0.7) is False
        assert is_low_confidence(0.9) is False

    def test_propagate_experimental(self):
        result = propagate_confidence([0.8], EvidenceType.DATO_EXPERIMENTAL)
        assert result == 0.8  # No penalty for experimental data

    def test_propagate_simulation(self):
        result = propagate_confidence([0.8], EvidenceType.RESULTADO_SIMULACION)
        assert result == pytest.approx(0.8 * 0.85)

    def test_propagate_inference(self):
        result = propagate_confidence([0.8], EvidenceType.INFERENCIA)
        assert result == pytest.approx(0.8 * 0.7)

    def test_propagate_extrapolation(self):
        result = propagate_confidence([0.8], EvidenceType.EXTRAPOLACION)
        assert result == pytest.approx(0.8 * 0.5)

    def test_propagate_sin_soporte_rejected(self):
        result = propagate_confidence([0.9], EvidenceType.SIN_SOPORTE)
        assert result == 0.0

    def test_propagate_multiple_sources(self):
        # Takes minimum of sources
        result = propagate_confidence([0.8, 0.6, 0.9], EvidenceType.DATO_EXPERIMENTAL)
        assert result == 0.6

    def test_propagate_capped_by_r_squared(self):
        result = propagate_confidence(
            [0.9], EvidenceType.RESULTADO_SIMULACION, model_r_squared=0.5
        )
        assert result == 0.5

    def test_combine_independent(self):
        result = combine_independent([0.8, 0.7])
        assert result == pytest.approx(0.56)

    def test_weighted_confidence(self):
        result = weighted_confidence([(0.8, 2.0), (0.4, 1.0)])
        assert result == pytest.approx(2.0 / 3.0)

    def test_audit_assertion_accepted(self):
        audit = audit_assertion(
            "DTC shows 200 pT sensitivity",
            [0.52],
            EvidenceType.DATO_EXPERIMENTAL,
            ["DTC sensor paper"],
        )
        assert audit["accepted"] is True
        assert audit["confidence"] == 0.52
        assert audit["evidence_type"] == "DATO_EXPERIMENTAL"

    def test_audit_sin_soporte_rejected(self):
        audit = audit_assertion(
            "DTC cures cancer",
            [0.9],
            EvidenceType.SIN_SOPORTE,
        )
        assert audit["accepted"] is False
        assert "rejection_reason" in audit

    def test_source_confidence_values(self):
        assert SOURCE_CONFIDENCE["dtc_sensor_paper"] == 0.52
        assert SOURCE_CONFIDENCE["bennett_brassard"] == 0.74

    def test_propagate_source_paper_through_simulation(self):
        """Source paper (0.52) → simulation → should be < 0.52."""
        result = propagate_confidence(
            [SOURCE_CONFIDENCE["dtc_sensor_paper"]],
            EvidenceType.RESULTADO_SIMULACION,
        )
        assert result < SOURCE_CONFIDENCE["dtc_sensor_paper"]
        assert result == pytest.approx(0.52 * 0.85)


class TestPrompts:
    """Verify all prompts exist and contain expected placeholders."""

    def test_system_prompt(self):
        assert "TempusSensor" in SYSTEM_PROMPT
        assert "confidence" in SYSTEM_PROMPT.lower()
        assert "0.52" in SYSTEM_PROMPT
        assert "0.74" in SYSTEM_PROMPT

    def test_percibir_prompt(self):
        assert "{papers_json}" in PERCIBIR_PROMPT
        assert "post-2023" not in PERCIBIR_PROMPT or "EXPERIMENTAL" in PERCIBIR_PROMPT

    def test_modelar_prompt(self):
        assert "{graph_summary}" in MODELAR_PROMPT
        assert "{new_results}" in MODELAR_PROMPT
        assert "white" in MODELAR_PROMPT.lower()

    def test_planificar_prompt(self):
        assert "{knowledge_state}" in PLANIFICAR_PROMPT
        assert "INCREMENTAL" in PLANIFICAR_PROMPT
        assert "COMBINATORIA" in PLANIFICAR_PROMPT
        assert "RADICAL" in PLANIFICAR_PROMPT
        assert "moonshot" in PLANIFICAR_PROMPT.lower()

    def test_sentir_prompt(self):
        assert "DALY" in SENTIR_PROMPT
        assert "{best_sensitivity_pT}" in SENTIR_PROMPT

    def test_decidir_prompt(self):
        assert "Pareto" in DECIDIR_PROMPT
        assert "robustness" in DECIDIR_PROMPT.lower() or "robustez" in DECIDIR_PROMPT.lower()

    def test_gobernar_prompt(self):
        assert "DATO_EXPERIMENTAL" in GOBERNAR_PROMPT
        assert "SIN_SOPORTE" in GOBERNAR_PROMPT
        assert "ETICO" in GOBERNAR_PROMPT

    def test_ejecutar_prompt(self):
        assert "{target_field_pT}" in EJECUTAR_PROMPT
        assert "false positive" in EJECUTAR_PROMPT.lower() or "falsos positivos" in EJECUTAR_PROMPT.lower()

    def test_recordar_prompt(self):
        assert "EPISODIC" in RECORDAR_PROMPT
        assert "SEMANTIC" in RECORDAR_PROMPT
        assert "PROCEDURAL" in RECORDAR_PROMPT
        assert "PROSPECTIVE" in RECORDAR_PROMPT

    def test_reflexionar_prompt(self):
        assert "calibration" in REFLEXIONAR_PROMPT.lower() or "calibración" in REFLEXIONAR_PROMPT.lower() or "ECE" in REFLEXIONAR_PROMPT
        assert "BENEFICENCIA" in REFLEXIONAR_PROMPT
        assert "NO_MALEFICENCIA" in REFLEXIONAR_PROMPT
        assert "JUSTICIA" in REFLEXIONAR_PROMPT
        assert "TRANSPARENCIA" in REFLEXIONAR_PROMPT


class TestAgentCycle:
    def test_all_nine_phases_defined(self):
        """Ensure all 9 phases exist in the orchestrator."""
        from src.agent.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        phase_methods = [
            "_phase_percibir",
            "_phase_modelar",
            "_phase_planificar",
            "_phase_sentir",
            "_phase_decidir",
            "_phase_gobernar",
            "_phase_ejecutar",
            "_phase_recordar",
            "_phase_reflexionar",
        ]
        for method in phase_methods:
            assert hasattr(orch, method), f"Missing phase method: {method}"

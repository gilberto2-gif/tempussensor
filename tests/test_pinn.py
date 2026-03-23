"""Tests for PINN physics and counterfactual analysis."""

import numpy as np
import pytest

from src.ml.pinn_physics import (
    CLINICAL_TARGETS,
    MATERIAL_PARAMS,
    compare_to_clinical,
    decoherence_rate,
    dtc_order_parameter,
    magnetic_susceptibility_dtc,
    theoretical_sensitivity,
)
from src.ml.counterfactual import (
    parameter_sweep_1d,
    pareto_frontier,
    robustness_analysis,
)


class TestPINNPhysics:
    def test_decoherence_rate_nv(self):
        rate = decoherence_rate("NV_DIAMOND", 300.0)
        assert rate > 0
        # Room temp should give faster decoherence
        rate_cold = decoherence_rate("NV_DIAMOND", 4.0)
        assert rate > rate_cold

    def test_decoherence_rate_superconductor_above_tc(self):
        tc = MATERIAL_PARAMS["SUPERCONDUCTOR"]["T_critical_K"]
        rate = decoherence_rate("SUPERCONDUCTOR", tc + 1)
        assert rate == 1e12  # Instant decoherence above Tc

    def test_dtc_order_parameter(self):
        # Create signal with subharmonic at drive_freq/2
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        drive_freq = 100.0
        signal = np.sin(2 * np.pi * drive_freq / 2 * t)  # Pure subharmonic

        order = dtc_order_parameter(signal, drive_freq, dt)
        assert 0.0 <= order <= 1.0
        assert order > 0.5  # Should detect the subharmonic

    def test_magnetic_susceptibility(self):
        chi = magnetic_susceptibility_dtc(
            "NV_DIAMOND", n_spins=15, temperature_k=300,
            drive_power_w=0.05, drive_freq_hz=10, order_param=0.8,
        )
        assert chi > 0

    def test_theoretical_sensitivity(self):
        sens = theoretical_sensitivity(
            "NV_DIAMOND", n_spins=15, temperature_k=300,
            drive_power_w=0.05, drive_freq_hz=10,
        )
        assert "0.5-10Hz" in sens
        assert "10-50Hz" in sens
        assert "50-100Hz" in sens
        assert sens["0.5-10Hz"] > 0
        # Low freq should be better (lower number)
        assert sens["0.5-10Hz"] <= sens["50-100Hz"]

    def test_compare_to_clinical(self):
        sens = theoretical_sensitivity(
            "NV_DIAMOND", n_spins=15, temperature_k=300,
            drive_power_w=0.05, drive_freq_hz=10,
        )
        comparison = compare_to_clinical(sens)
        assert "MEG" in comparison
        assert "MCG" in comparison
        assert "BIOMARCADORES" in comparison
        for target in comparison.values():
            assert "target_pT" in target
            assert "current_pT" in target
            assert "gap_factor" in target
            assert "meets_target" in target


class TestCounterfactual:
    def test_parameter_sweep_1d(self):
        sweep = parameter_sweep_1d(
            "NV_DIAMOND",
            "temperatura_k",
            np.linspace(1, 400, 20),
            fixed_params={"n_spins": 15},
        )
        assert len(sweep["param_values"]) == 20
        assert len(sweep["sensitivity_10_50"]) == 20
        assert "optimal" in sweep
        assert all(s > 0 for s in sweep["sensitivity_10_50"])

    def test_pareto_frontier(self):
        result = pareto_frontier(n_samples=50)
        assert result["n_pareto"] > 0
        assert len(result["pareto_optimal"]) <= 50
        for p in result["pareto_optimal"]:
            assert "material" in p
            assert "sensitivity_pT" in p
            assert "cost_usd" in p

    def test_robustness(self):
        result = robustness_analysis(
            "NV_DIAMOND",
            {"n_spins": 15, "temperatura_k": 300, "potencia_laser_w": 0.05, "frecuencia_hz": 10},
            variation_pct=5.0,
            n_samples=50,
        )
        assert result["mean"] > 0
        assert result["std"] > 0
        assert result["cv"] > 0
        assert "robust" in result

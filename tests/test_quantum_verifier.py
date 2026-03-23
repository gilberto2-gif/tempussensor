"""Tests for BB84-inspired quantum integrity verification."""

import numpy as np
import pytest

from src.agent.quantum_verifier import (
    CertificationLevel,
    QuantumVerifier,
    TestState,
)


class TestQuantumVerifier:
    def setup_method(self):
        self.verifier = QuantumVerifier()

    def test_verify_clean_signal(self):
        """A clean DTC signal should pass all tests."""
        dt = 0.001
        t = np.arange(0, 2.0, dt)
        drive_freq = 100.0
        # Clean subharmonic signal + some noise
        signal = (
            np.sin(2 * np.pi * drive_freq / 2 * t) * 5
            + np.random.normal(0, 0.1, len(t))
        )

        result = self.verifier.verify(signal, drive_freq, dt, n_spins=10)

        assert result.hash_datos != ""
        assert result.certificacion in [
            CertificationLevel.INTEGRO,
            CertificationLevel.CON_ADVERTENCIA,
        ]

    def test_verify_noisy_signal(self):
        """A very noisy signal should fail or warn."""
        signal = np.random.normal(0, 10, 2000)
        result = self.verifier.verify(signal, 100.0, 0.001, n_spins=10)

        # Should not be INTEGRO with pure noise
        assert result.certificacion != CertificationLevel.INTEGRO

    def test_coherence_witness(self):
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        drive_freq = 50.0
        signal = np.sin(2 * np.pi * drive_freq / 2 * t) * 10

        result = self.verifier._test_coherence(signal, drive_freq, dt)
        assert result.fidelidad_dtc >= 0
        assert result.estado in [TestState.PASS, TestState.WARN, TestState.FAIL]

    def test_correlation_test(self):
        signal = np.sin(np.linspace(0, 20 * np.pi, 2000))
        result = self.verifier._test_correlations(signal, n_spins=10)
        assert result.correlacion_media >= 0
        assert result.estado in [TestState.PASS, TestState.WARN, TestState.FAIL]

    def test_noise_fingerprint(self):
        signal = np.random.normal(0, 1, 1000)
        result = self.verifier._test_noise_fingerprint(signal, dt=0.001)
        assert result.divergencia_kl >= 0
        assert isinstance(result.fuentes_interferencia, list)

    def test_noise_fingerprint_with_expected(self):
        signal = np.random.normal(0, 1, 1000)
        # Expected spectrum matches white noise
        expected = np.ones(501)  # rfft of 1000 points
        result = self.verifier._test_noise_fingerprint(signal, dt=0.001, expected_spectrum=expected)
        assert result.divergencia_kl >= 0

    def test_verify_simulation(self):
        result = self.verifier.verify_simulation(
            predicted_sensitivity=100.0,
            model_r_squared=0.9,
            training_loss=0.01,
        )
        assert result.certificacion == CertificationLevel.INTEGRO
        assert result.hash_datos != ""

    def test_verify_simulation_bad_model(self):
        result = self.verifier.verify_simulation(
            predicted_sensitivity=100.0,
            model_r_squared=0.3,  # Bad R²
            training_loss=5.0,    # High loss
        )
        # Should not be INTEGRO with bad model
        assert result.certificacion in [
            CertificationLevel.CON_ADVERTENCIA,
            CertificationLevel.NO_CONFIABLE,
        ]

    def test_data_hash_deterministic(self):
        signal = np.array([1.0, 2.0, 3.0])
        r1 = self.verifier.verify(signal, 100.0, 0.001, n_spins=2)
        r2 = self.verifier.verify(signal, 100.0, 0.001, n_spins=2)
        assert r1.hash_datos == r2.hash_datos

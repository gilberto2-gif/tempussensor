"""Capa 8 — Quantum integrity verification inspired by BB84.

Verifies DTC sensor data integrity using three tests:
1. Coherence witness: DTC order parameter fidelity during acquisition
2. Quantum correlations: spin-spin correlation function — below threshold = classical interference
3. Noise fingerprint: KL divergence between measured vs expected noise spectrum

Certification: INTEGRO | CON_ADVERTENCIA | NO_CONFIABLE

This PREVENTS false positives from noise that could lead to erroneous diagnoses.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CertificationLevel(str, Enum):
    INTEGRO = "INTEGRO"
    CON_ADVERTENCIA = "CON_ADVERTENCIA"
    NO_CONFIABLE = "NO_CONFIABLE"


class TestState(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CoherenceResult:
    """Coherence witness: fidelity of DTC order parameter."""
    fidelidad_dtc: float
    umbral: float = 0.8
    estado: TestState = TestState.PASS

    def __post_init__(self):
        if self.fidelidad_dtc >= self.umbral:
            self.estado = TestState.PASS
        elif self.fidelidad_dtc >= self.umbral * 0.7:
            self.estado = TestState.WARN
        else:
            self.estado = TestState.FAIL


@dataclass
class CorrelationResult:
    """Quantum correlations: spin-spin correlation function."""
    correlacion_media: float
    umbral: float = 0.6
    estado: TestState = TestState.PASS

    def __post_init__(self):
        if self.correlacion_media >= self.umbral:
            self.estado = TestState.PASS
        elif self.correlacion_media >= self.umbral * 0.7:
            self.estado = TestState.WARN
        else:
            self.estado = TestState.FAIL


@dataclass
class NoiseResult:
    """Noise fingerprint: KL divergence from expected spectrum."""
    divergencia_kl: float
    umbral: float = 0.5
    estado: TestState = TestState.PASS
    fuentes_interferencia: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.divergencia_kl <= self.umbral:
            self.estado = TestState.PASS
        elif self.divergencia_kl <= self.umbral * 1.5:
            self.estado = TestState.WARN
        else:
            self.estado = TestState.FAIL


@dataclass
class IntegrityVerification:
    """Complete integrity verification result."""
    coherencia: CoherenceResult
    correlaciones: CorrelationResult
    ruido: NoiseResult
    certificacion: CertificationLevel = CertificationLevel.INTEGRO
    hash_datos: str = ""
    version_protocolo: str = "1.0"

    def __post_init__(self):
        states = [self.coherencia.estado, self.correlaciones.estado, self.ruido.estado]
        if any(s == TestState.FAIL for s in states):
            self.certificacion = CertificationLevel.NO_CONFIABLE
        elif any(s == TestState.WARN for s in states):
            self.certificacion = CertificationLevel.CON_ADVERTENCIA
        else:
            self.certificacion = CertificationLevel.INTEGRO


class QuantumVerifier:
    """BB84-inspired integrity verifier for DTC sensor data."""

    def __init__(
        self,
        coherence_threshold: float = 0.8,
        correlation_threshold: float = 0.6,
        noise_threshold: float = 0.5,
    ):
        self.coherence_threshold = coherence_threshold
        self.correlation_threshold = correlation_threshold
        self.noise_threshold = noise_threshold

    def verify(
        self,
        time_series: np.ndarray,
        drive_frequency: float,
        dt: float,
        expected_noise_spectrum: np.ndarray | None = None,
        n_spins: int = 10,
    ) -> IntegrityVerification:
        """Run full integrity verification on sensor data.

        Args:
            time_series: Raw sensor signal (time domain)
            drive_frequency: DTC driving frequency (Hz)
            dt: Sampling interval (seconds)
            expected_noise_spectrum: Expected noise PSD (if known)
            n_spins: Number of spins in sensor

        Returns:
            IntegrityVerification with all test results and certification.
        """
        # Hash the data for traceability
        data_hash = hashlib.sha256(time_series.tobytes()).hexdigest()

        # Test 1: Coherence witness
        coherence = self._test_coherence(time_series, drive_frequency, dt)

        # Test 2: Quantum correlations
        correlations = self._test_correlations(time_series, n_spins)

        # Test 3: Noise fingerprint
        noise = self._test_noise_fingerprint(time_series, dt, expected_noise_spectrum)

        result = IntegrityVerification(
            coherencia=coherence,
            correlaciones=correlations,
            ruido=noise,
            hash_datos=data_hash,
        )

        logger.info(
            "integrity_verified",
            certificacion=result.certificacion.value,
            coherencia=coherence.estado.value,
            correlaciones=correlations.estado.value,
            ruido=noise.estado.value,
        )

        return result

    def _test_coherence(
        self,
        signal: np.ndarray,
        drive_freq: float,
        dt: float,
    ) -> CoherenceResult:
        """Test 1: DTC order parameter fidelity.

        Checks that the subharmonic response at ω_drive/2 is stable
        and above threshold throughout the acquisition window.
        """
        n = len(signal)
        # Split into windows and check consistency
        window_size = min(n // 4, 1000)
        if window_size < 10:
            return CoherenceResult(fidelidad_dtc=0.0, umbral=self.coherence_threshold)

        fidelities = []
        for start in range(0, n - window_size, window_size):
            window = signal[start : start + window_size]
            freqs = np.fft.rfftfreq(len(window), d=dt)
            fft = np.abs(np.fft.rfft(window))

            # Find subharmonic at drive_freq/2
            target = drive_freq / 2.0
            idx = np.argmin(np.abs(freqs - target))

            # Fidelity = subharmonic power / total power
            sub_power = fft[idx] ** 2
            total_power = np.sum(fft**2) + 1e-30
            fidelities.append(sub_power / total_power)

        mean_fidelity = float(np.mean(fidelities)) if fidelities else 0.0
        # Normalize to 0-1 range
        normalized = min(1.0, mean_fidelity * 10)  # Scale factor for typical signals

        return CoherenceResult(
            fidelidad_dtc=normalized,
            umbral=self.coherence_threshold,
        )

    def _test_correlations(
        self,
        signal: np.ndarray,
        n_spins: int,
    ) -> CorrelationResult:
        """Test 2: Spin-spin correlation function.

        Simulates correlation check: if spatial correlations between
        spin signals fall below threshold, likely classical interference.
        """
        n = len(signal)
        if n < 2 * n_spins:
            return CorrelationResult(correlacion_media=0.0, umbral=self.correlation_threshold)

        # Auto-correlation as proxy for spin-spin correlations
        # (in real implementation, would use multi-channel data)
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-30)
        autocorr = np.correlate(signal_norm[:1000], signal_norm[:1000], mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-30)

        # Check correlation at lag = n_spins (proxy for spatial correlation)
        lag = min(n_spins, len(autocorr) - 1)
        corr_at_lag = abs(autocorr[lag])

        # Average of first n_spins lags
        avg_corr = float(np.mean(np.abs(autocorr[1 : min(n_spins + 1, len(autocorr))])))

        return CorrelationResult(
            correlacion_media=avg_corr,
            umbral=self.correlation_threshold,
        )

    def _test_noise_fingerprint(
        self,
        signal: np.ndarray,
        dt: float,
        expected_spectrum: np.ndarray | None = None,
    ) -> NoiseResult:
        """Test 3: KL divergence between measured and expected noise spectrum.

        If expected spectrum not provided, compares against theoretical
        white + 1/f noise model.
        """
        # Compute PSD of measured signal
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        measured_psd = np.abs(np.fft.rfft(signal)) ** 2 / n

        if expected_spectrum is None:
            # Generate expected: white noise + 1/f component
            expected_psd = np.ones_like(measured_psd)
            # Add 1/f component
            with np.errstate(divide="ignore", invalid="ignore"):
                f_nonzero = np.where(freqs > 0, freqs, 1.0)
                expected_psd += 1.0 / f_nonzero
            expected_psd[0] = expected_psd[1]  # Fix DC
        else:
            expected_psd = expected_spectrum
            if len(expected_psd) != len(measured_psd):
                expected_psd = np.interp(
                    np.linspace(0, 1, len(measured_psd)),
                    np.linspace(0, 1, len(expected_psd)),
                    expected_psd,
                )

        # Normalize to probability distributions
        measured_norm = measured_psd / (np.sum(measured_psd) + 1e-30)
        expected_norm = expected_psd / (np.sum(expected_psd) + 1e-30)

        # KL divergence (with smoothing to avoid log(0))
        epsilon = 1e-10
        measured_smooth = measured_norm + epsilon
        expected_smooth = expected_norm + epsilon
        measured_smooth /= np.sum(measured_smooth)
        expected_smooth /= np.sum(expected_smooth)

        kl_div = float(np.sum(measured_smooth * np.log(measured_smooth / expected_smooth)))

        # Identify interference sources (peaks in residual)
        residual = measured_norm - expected_norm
        peak_indices = np.where(residual > 3 * np.std(residual))[0]
        interference_sources = []
        for idx in peak_indices[:5]:  # Top 5
            if idx < len(freqs):
                interference_sources.append(f"{freqs[idx]:.1f}Hz")

        return NoiseResult(
            divergencia_kl=kl_div,
            umbral=self.noise_threshold,
            fuentes_interferencia=interference_sources,
        )

    def verify_simulation(
        self,
        predicted_sensitivity: float,
        model_r_squared: float,
        training_loss: float,
    ) -> IntegrityVerification:
        """Simplified verification for simulation results (no raw signal)."""
        # Coherence: based on model quality
        coherence = CoherenceResult(
            fidelidad_dtc=model_r_squared,
            umbral=self.coherence_threshold,
        )

        # Correlations: based on training convergence
        convergence = 1.0 / (1.0 + training_loss)
        correlations = CorrelationResult(
            correlacion_media=convergence,
            umbral=self.correlation_threshold,
        )

        # Noise: based on prediction reasonableness
        # Sensitivity should be positive and within physical bounds
        is_reasonable = 0.01 < predicted_sensitivity < 1e6
        noise = NoiseResult(
            divergencia_kl=0.1 if is_reasonable else 2.0,
            umbral=self.noise_threshold,
        )

        data_hash = hashlib.sha256(
            f"{predicted_sensitivity}:{model_r_squared}:{training_loss}".encode()
        ).hexdigest()

        return IntegrityVerification(
            coherencia=coherence,
            correlaciones=correlations,
            ruido=noise,
            hash_datos=data_hash,
        )

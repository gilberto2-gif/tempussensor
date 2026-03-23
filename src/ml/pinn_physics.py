"""Physical equations and constraints for DTC PINN simulation.

Contains:
- Schrödinger equation components for periodically driven systems
- DTC order parameter definitions
- Magnetic susceptibility relations
- Thermal decoherence models
- Constraint functions for PINN training
"""

import numpy as np


# ===========================================================================
# Physical constants
# ===========================================================================

HBAR = 1.054571817e-34          # J·s
K_BOLTZMANN = 1.380649e-23      # J/K
MU_BOHR = 9.2740100783e-24      # J/T
GAMMA_ELECTRON = 1.76085963e11  # rad/(s·T), electron gyromagnetic ratio
GAMMA_NV = 2.8e10               # Hz/T, NV center
GAMMA_YB = 2.1e10               # Hz/T, Yb+ ion


# ===========================================================================
# Material-specific parameters
# ===========================================================================

MATERIAL_PARAMS = {
    "NV_DIAMOND": {
        "gamma": GAMMA_NV,                # Gyromagnetic ratio (Hz/T)
        "zero_field_splitting": 2.87e9,   # Hz
        "T2_room_temp_us": 1.0,          # Coherence time at 300K (μs)
        "T2_cryo_us": 100.0,             # Coherence time at 4K (μs)
        "optimal_drive_freq_ghz": 2.87,  # GHz
        "drive_type": "MICROWAVE",
        "detection": "fluorescence",
    },
    "TRAPPED_ION": {
        "gamma": GAMMA_YB,
        "hyperfine_splitting": 12.6e9,    # Hz (Yb+)
        "T2_us": 1000.0,                 # Long coherence in ion trap
        "trap_freq_mhz": 1.0,            # Paul trap frequency
        "laser_wavelength_nm": 369.5,    # S-P transition Yb+
        "drive_type": "LASER",
        "detection": "fluorescence_resolved",
    },
    "SUPERCONDUCTOR": {
        "gamma": GAMMA_ELECTRON,
        "T_critical_K": 15.0,            # Critical temperature
        "T2_us": 50.0,                   # At operating T
        "resonance_ghz": 5.0,            # Transmon frequency
        "drive_type": "MICROWAVE",
        "detection": "dispersive_readout",
    },
}


# ===========================================================================
# DTC Order Parameter
# ===========================================================================


def dtc_order_parameter(
    signal: np.ndarray,
    drive_freq: float,
    dt: float,
) -> float:
    """Compute the DTC order parameter from a time-domain signal.

    The DTC order parameter measures the subharmonic response:
    it's the Fourier amplitude at ω_drive/2 (period doubling).

    Args:
        signal: Time-domain signal (e.g., magnetization)
        drive_freq: Driving frequency in Hz
        dt: Time step in seconds

    Returns:
        Order parameter amplitude (0 = no DTC, 1 = perfect DTC)
    """
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.abs(np.fft.rfft(signal)) / n

    # Find subharmonic peak at drive_freq/2
    target_freq = drive_freq / 2.0
    idx = np.argmin(np.abs(freqs - target_freq))

    # Normalize by total power
    subharmonic_power = fft_vals[idx] ** 2
    total_power = np.sum(fft_vals**2) + 1e-30

    return float(np.sqrt(subharmonic_power / total_power))


# ===========================================================================
# Decoherence models
# ===========================================================================


def decoherence_rate(
    material_type: str,
    temperature_k: float,
) -> float:
    """Compute decoherence rate Γ = 1/T2 in Hz.

    Models temperature-dependent decoherence for each material.
    """
    params = MATERIAL_PARAMS.get(material_type, MATERIAL_PARAMS["NV_DIAMOND"])

    if material_type == "NV_DIAMOND":
        t2_room = params["T2_room_temp_us"]
        t2_cryo = params["T2_cryo_us"]
        # Interpolate T2 between cryo and room temp
        frac = min(temperature_k / 300.0, 1.0)
        t2_us = t2_cryo * (1 - frac) + t2_room * frac
        return 1e6 / t2_us  # Convert μs to Hz

    elif material_type == "TRAPPED_ION":
        # Ions in vacuum: weak T dependence (heating from electrodes)
        t2 = params["T2_us"] / (1 + 0.0001 * temperature_k)
        return 1e6 / t2

    elif material_type == "SUPERCONDUCTOR":
        tc = params["T_critical_K"]
        if temperature_k >= tc:
            return 1e12  # Above Tc: instant decoherence
        t2 = params["T2_us"] * (1 - (temperature_k / tc) ** 2)
        return 1e6 / max(t2, 0.01)

    return 1e6  # Default fallback


# ===========================================================================
# Magnetic susceptibility
# ===========================================================================


def magnetic_susceptibility_dtc(
    material_type: str,
    n_spins: int,
    temperature_k: float,
    drive_power_w: float,
    drive_freq_hz: float,
    order_param: float,
) -> float:
    """Compute magnetic susceptibility χ of DTC sensor.

    χ_DTC ∝ N × γ² × T2 × η_DTC / (k_B × T)

    where η_DTC is the DTC order parameter enhancement factor.

    Args:
        material_type: Material system
        n_spins: Number of spins/qubits
        temperature_k: Temperature in K
        drive_power_w: Driving power in W
        drive_freq_hz: Driving frequency in Hz
        order_param: DTC order parameter (0-1)

    Returns:
        Magnetic susceptibility (dimensionless SI)
    """
    params = MATERIAL_PARAMS.get(material_type, MATERIAL_PARAMS["NV_DIAMOND"])
    gamma = params["gamma"]

    # Base Curie susceptibility
    if temperature_k < 0.01:
        temperature_k = 0.01  # Avoid division by zero

    chi_curie = n_spins * MU_BOHR**2 * gamma**2 / (3 * K_BOLTZMANN * temperature_k)

    # DTC enhancement: order parameter amplifies response at subharmonic
    # This is the key mechanism — the DTC acts as a frequency converter
    # that concentrates signal energy at the detection frequency
    dtc_enhancement = 1.0 + 10.0 * order_param**2

    # Drive power efficiency
    power_factor = np.tanh(drive_power_w * 10)

    # Coherence factor
    gamma_dec = decoherence_rate(material_type, temperature_k)
    coherence_factor = drive_freq_hz / (drive_freq_hz + gamma_dec)

    return chi_curie * dtc_enhancement * power_factor * coherence_factor


# ===========================================================================
# Sensitivity estimation
# ===========================================================================


def theoretical_sensitivity(
    material_type: str,
    n_spins: int,
    temperature_k: float,
    drive_power_w: float,
    drive_freq_hz: float,
    measurement_time_s: float = 1.0,
) -> dict:
    """Compute theoretical sensitivity of DTC sensor.

    Returns sensitivity in pT/√Hz for each frequency band.

    Based on:
    δB = 1/(γ × √N × T2_eff × √(t_meas)) × noise_factor

    Enhanced by DTC order parameter at the subharmonic.
    """
    params = MATERIAL_PARAMS.get(material_type, MATERIAL_PARAMS["NV_DIAMOND"])
    gamma = params["gamma"]
    gamma_dec = decoherence_rate(material_type, temperature_k)
    t2_eff = 1.0 / gamma_dec if gamma_dec > 0 else 1.0

    # Base sensitivity (T/√Hz)
    base = 1.0 / (gamma * np.sqrt(n_spins) * t2_eff * np.sqrt(measurement_time_s))

    # DTC enhancement (estimated order parameter)
    # Better driving → better order parameter → better sensitivity
    drive_efficiency = np.tanh(drive_power_w * 5)
    est_order_param = drive_efficiency * np.exp(-gamma_dec / drive_freq_hz)
    dtc_factor = 1.0 / (1.0 + 10.0 * est_order_param**2)

    # Thermal noise floor
    thermal_factor = np.sqrt(K_BOLTZMANN * temperature_k / (MU_BOHR * max(n_spins, 1)))

    sensitivity_t = base * dtc_factor * (1 + 0.01 * thermal_factor)

    # Convert to pT/√Hz
    sens_pt = sensitivity_t * 1e12

    # Frequency-band dependent (higher freq → more noise)
    return {
        "0.5-10Hz": sens_pt * 0.8,
        "10-50Hz": sens_pt * 1.0,
        "50-100Hz": sens_pt * 1.5,
        "overall_pT": sens_pt,
        "estimated_order_param": est_order_param,
        "T2_eff_us": t2_eff * 1e6,
        "decoherence_rate_hz": gamma_dec,
    }


# ===========================================================================
# Clinical target comparison
# ===========================================================================

CLINICAL_TARGETS = {
    "MEG": {"sensitivity_fT": 10, "freq_range_hz": (1, 100), "description": "Magnetoencephalography"},
    "MCG": {"sensitivity_fT": 50, "freq_range_hz": (0.1, 50), "description": "Magnetocardiography"},
    "BIOMARCADORES": {"sensitivity_pT": 1, "freq_range_hz": (0.01, 10), "description": "Magnetic biomarkers"},
}


def compare_to_clinical(sensitivity_pt_per_band: dict) -> dict:
    """Compare DTC sensor sensitivity to clinical requirements.

    Returns gap analysis: how far from each clinical target.
    """
    results = {}
    for target_name, target in CLINICAL_TARGETS.items():
        if "sensitivity_fT" in target:
            target_pt = target["sensitivity_fT"] / 1000.0  # fT → pT
        else:
            target_pt = target["sensitivity_pT"]

        # Find the relevant band
        freq_lo, freq_hi = target["freq_range_hz"]
        relevant_sens = []
        for band, sens in sensitivity_pt_per_band.items():
            if band == "overall_pT":
                continue
            # Parse band
            parts = band.replace("Hz", "").split("-")
            if len(parts) == 2:
                blo, bhi = float(parts[0]), float(parts[1])
                if blo <= freq_hi and bhi >= freq_lo:
                    relevant_sens.append(sens)

        current_sens = min(relevant_sens) if relevant_sens else sensitivity_pt_per_band.get("overall_pT", 999)

        gap_factor = current_sens / target_pt if target_pt > 0 else float("inf")
        results[target_name] = {
            "target_pT": target_pt,
            "current_pT": current_sens,
            "gap_factor": gap_factor,
            "meets_target": gap_factor <= 1.0,
            "orders_of_magnitude_gap": np.log10(max(gap_factor, 1e-10)),
        }

    return results

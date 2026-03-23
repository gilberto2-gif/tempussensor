"""Counterfactual simulation: parameter sweeps, Pareto fronts, and robustness analysis.

Implements Phase 5 (DECIDIR) of the agent loop:
- 1D and 2D parameter sweeps via PINN
- Pareto frontier: sensitivity vs temperature vs cost
- Robustness analysis: ±5% parameter variation
- Bifurcation detection in parameter space
"""

import itertools

import numpy as np
import structlog

from src.ml.pinn_physics import (
    MATERIAL_PARAMS,
    compare_to_clinical,
    theoretical_sensitivity,
)

logger = structlog.get_logger(__name__)


def parameter_sweep_1d(
    material_type: str,
    sweep_param: str,
    sweep_range: np.ndarray,
    fixed_params: dict,
) -> dict:
    """1D parameter sweep using theoretical model.

    Args:
        material_type: DTC material system
        sweep_param: Parameter to vary (temperatura_k, potencia_laser_w, n_spins, etc.)
        sweep_range: Array of values to sweep
        fixed_params: Dict of fixed parameters

    Returns:
        Dict with sweep results including sensitivities per band.
    """
    results = {
        "param_name": sweep_param,
        "param_values": sweep_range.tolist(),
        "sensitivity_05_10": [],
        "sensitivity_10_50": [],
        "sensitivity_50_100": [],
        "order_param": [],
        "t2_eff_us": [],
    }

    base = {
        "n_spins": fixed_params.get("n_spins", 15),
        "temperature_k": fixed_params.get("temperatura_k", 300.0),
        "drive_power_w": fixed_params.get("potencia_laser_w", 0.05),
        "drive_freq_hz": fixed_params.get("frecuencia_hz", 10.0),
    }

    param_map = {
        "temperatura_k": "temperature_k",
        "potencia_laser_w": "drive_power_w",
        "n_spins": "n_spins",
        "frecuencia_hz": "drive_freq_hz",
    }

    mapped_param = param_map.get(sweep_param, sweep_param)

    for val in sweep_range:
        params = {**base, mapped_param: float(val)}
        sens = theoretical_sensitivity(material_type=material_type, **params)

        results["sensitivity_05_10"].append(sens["0.5-10Hz"])
        results["sensitivity_10_50"].append(sens["10-50Hz"])
        results["sensitivity_50_100"].append(sens["50-100Hz"])
        results["order_param"].append(sens["estimated_order_param"])
        results["t2_eff_us"].append(sens["T2_eff_us"])

    # Detect optimal point
    best_idx = int(np.argmin(results["sensitivity_10_50"]))
    results["optimal"] = {
        "param_value": results["param_values"][best_idx],
        "best_sensitivity_10_50": results["sensitivity_10_50"][best_idx],
    }

    # Detect bifurcation points (sharp changes in sensitivity)
    sens_arr = np.array(results["sensitivity_10_50"])
    gradient = np.gradient(sens_arr)
    grad_changes = np.abs(np.gradient(gradient))
    threshold = 3.0 * np.std(grad_changes)
    bifurcation_indices = np.where(grad_changes > threshold)[0].tolist()
    results["bifurcation_points"] = [
        {"index": int(i), "param_value": results["param_values"][i]}
        for i in bifurcation_indices
    ]

    return results


def parameter_sweep_2d(
    material_type: str,
    param1: str,
    range1: np.ndarray,
    param2: str,
    range2: np.ndarray,
    fixed_params: dict,
) -> dict:
    """2D parameter sweep — creates a heatmap of sensitivity."""
    sensitivity_grid = np.zeros((len(range1), len(range2)))

    base = {
        "n_spins": fixed_params.get("n_spins", 15),
        "temperature_k": fixed_params.get("temperatura_k", 300.0),
        "drive_power_w": fixed_params.get("potencia_laser_w", 0.05),
        "drive_freq_hz": fixed_params.get("frecuencia_hz", 10.0),
    }

    param_map = {
        "temperatura_k": "temperature_k",
        "potencia_laser_w": "drive_power_w",
        "n_spins": "n_spins",
        "frecuencia_hz": "drive_freq_hz",
    }

    m1, m2 = param_map.get(param1, param1), param_map.get(param2, param2)

    for i, v1 in enumerate(range1):
        for j, v2 in enumerate(range2):
            params = {**base, m1: float(v1), m2: float(v2)}
            sens = theoretical_sensitivity(material_type=material_type, **params)
            sensitivity_grid[i, j] = sens["10-50Hz"]

    # Find global optimum
    opt_idx = np.unravel_index(np.argmin(sensitivity_grid), sensitivity_grid.shape)

    return {
        "param1": param1,
        "param2": param2,
        "range1": range1.tolist(),
        "range2": range2.tolist(),
        "sensitivity_grid": sensitivity_grid.tolist(),
        "optimal": {
            param1: float(range1[opt_idx[0]]),
            param2: float(range2[opt_idx[1]]),
            "sensitivity_pT": float(sensitivity_grid[opt_idx]),
        },
    }


def pareto_frontier(
    materials: list[str] | None = None,
    n_samples: int = 1000,
) -> dict:
    """Compute Pareto frontier: sensitivity vs temperature vs cost.

    Samples the parameter space and identifies non-dominated solutions.
    Objectives (all minimize):
    1. Sensitivity (pT/√Hz) at 10-50Hz
    2. Temperature (K) — lower is harder/costlier
    3. Estimated cost (USD)
    """
    if materials is None:
        materials = list(MATERIAL_PARAMS.keys())

    rng = np.random.default_rng(42)
    points = []

    for _ in range(n_samples):
        mat = rng.choice(materials)
        n_spins = int(rng.integers(5, 100))
        temp_k = rng.uniform(0.01, 400.0)
        power_w = rng.uniform(0.001, 1.0)
        freq_hz = rng.uniform(1.0, 100.0)

        sens = theoretical_sensitivity(mat, n_spins, temp_k, power_w, freq_hz)
        cost = _estimate_cost(mat, n_spins, temp_k)

        points.append({
            "material": mat,
            "n_spins": n_spins,
            "temperatura_k": temp_k,
            "potencia_w": power_w,
            "frecuencia_hz": freq_hz,
            "sensitivity_pT": sens["10-50Hz"],
            "cost_usd": cost,
            "clinical_comparison": compare_to_clinical(sens),
        })

    # Find Pareto-optimal points
    objectives = np.array([
        [p["sensitivity_pT"], -p["temperatura_k"], p["cost_usd"]]
        for p in points
    ])
    pareto_mask = _is_pareto_optimal(objectives)
    pareto_points = [p for p, m in zip(points, pareto_mask) if m]

    logger.info("pareto_computed", total_points=len(points), pareto_points=len(pareto_points))

    return {
        "all_points": points,
        "pareto_optimal": pareto_points,
        "n_pareto": len(pareto_points),
    }


def robustness_analysis(
    material_type: str,
    nominal_params: dict,
    variation_pct: float = 5.0,
    n_samples: int = 200,
) -> dict:
    """Robustness analysis: how sensitive is the output to ±variation_pct% input changes.

    Monte Carlo: sample each parameter within ±variation_pct% of nominal.
    """
    rng = np.random.default_rng(42)
    param_names = ["n_spins", "temperatura_k", "potencia_laser_w", "frecuencia_hz"]

    base = {
        "n_spins": nominal_params.get("n_spins", 15),
        "temperature_k": nominal_params.get("temperatura_k", 300.0),
        "drive_power_w": nominal_params.get("potencia_laser_w", 0.05),
        "drive_freq_hz": nominal_params.get("frecuencia_hz", 10.0),
    }

    param_map = {
        "n_spins": "n_spins",
        "temperatura_k": "temperature_k",
        "potencia_laser_w": "drive_power_w",
        "frecuencia_hz": "drive_freq_hz",
    }

    sensitivities = []
    for _ in range(n_samples):
        perturbed = {}
        for pname in param_names:
            nominal = nominal_params.get(pname, base[param_map[pname]])
            delta = nominal * variation_pct / 100.0
            val = rng.uniform(nominal - delta, nominal + delta)
            if pname == "n_spins":
                val = max(1, int(round(val)))
            else:
                val = max(0.001, float(val))
            perturbed[param_map[pname]] = val

        sens = theoretical_sensitivity(material_type=material_type, **perturbed)
        sensitivities.append(sens["10-50Hz"])

    arr = np.array(sensitivities)
    return {
        "nominal_sensitivity": theoretical_sensitivity(material_type=material_type, **base)["10-50Hz"],
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "cv": float(np.std(arr) / np.mean(arr)),  # Coefficient of variation
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "robust": bool(np.std(arr) / np.mean(arr) < 0.1),  # CV < 10% = robust
        "variation_pct": variation_pct,
        "n_samples": n_samples,
    }


def _estimate_cost(material: str, n_spins: int, temp_k: float) -> float:
    """Rough cost estimate in USD based on material and requirements."""
    base_costs = {
        "NV_DIAMOND": 5000,
        "TRAPPED_ION": 50000,
        "SUPERCONDUCTOR": 100000,
    }
    cost = base_costs.get(material, 10000)

    # More spins = more cost
    cost *= 1 + 0.01 * n_spins

    # Cryogenics cost: cheaper near room temp
    if temp_k < 4:
        cost += 200000  # Dilution fridge
    elif temp_k < 77:
        cost += 50000   # LN2/LHe cryostat
    elif temp_k < 250:
        cost += 10000   # Peltier cooling

    return cost


def _is_pareto_optimal(costs: np.ndarray) -> np.ndarray:
    """Find Pareto-optimal points (minimize all objectives)."""
    n = costs.shape[0]
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_optimal[i]:
            continue
        for j in range(n):
            if i == j or not is_optimal[j]:
                continue
            # j dominates i if j <= i in all and j < i in at least one
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_optimal[i] = False
                break

    return is_optimal

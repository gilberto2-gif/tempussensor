"""Dataset of crystal graphs for DTC sensor GNN.

Each sample represents a DTC configuration with:
- Node features: atom/spin properties
- Edge features: coupling parameters
- Global features: temperature, laser power, external field
- Target: sensitivity (pT/√Hz) per frequency band
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class DTCGraphDataset(InMemoryDataset):
    """In-memory dataset of DTC crystal graphs.

    Can be populated from:
    1. Extracted paper parameters (real data)
    2. Physics-based synthetic generation (for bootstrap training)
    """

    def __init__(self, root="data/dtc_graphs", transform=None):
        super().__init__(root, transform)
        self._data_list: list[Data] = []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def add_sample(self, data: Data):
        self._data_list.append(data)

    def add_from_parameters(
        self,
        n_spins: int,
        material_type: str,
        temperatura_k: float,
        potencia_laser_w: float,
        campo_externo_t: float,
        densidad_defectos: float,
        target_sensitivity: list[float] | None = None,
        j_coupling: float = 1.0,
    ):
        """Build a crystal graph from physical parameters."""
        data = build_crystal_graph(
            n_spins=n_spins,
            material_type=material_type,
            temperatura_k=temperatura_k,
            potencia_laser_w=potencia_laser_w,
            campo_externo_t=campo_externo_t,
            densidad_defectos=densidad_defectos,
            target_sensitivity=target_sensitivity,
            j_coupling=j_coupling,
        )
        self._data_list.append(data)


# Material type encoding
MATERIAL_ENCODING = {
    "NV_DIAMOND": 0,
    "TRAPPED_ION": 1,
    "SUPERCONDUCTOR": 2,
    "OTHER": 3,
}


def build_crystal_graph(
    n_spins: int = 10,
    material_type: str = "NV_DIAMOND",
    temperatura_k: float = 300.0,
    potencia_laser_w: float = 0.01,
    campo_externo_t: float = 0.0,
    densidad_defectos: float = 0.1,
    target_sensitivity: list[float] | None = None,
    j_coupling: float = 1.0,
) -> Data:
    """Create a PyG Data object representing a DTC crystal configuration.

    Node features [4]: atom_type_enc, local_field, resonance_freq, spin_value
    Edge features [3]: J_coupling, distance, interaction_type_enc
    Global [3]: temperature, laser_power, external_field
    Target [3]: sensitivity per band
    """
    mat_enc = MATERIAL_ENCODING.get(material_type, 3)
    rng = np.random.default_rng(hash((n_spins, material_type, temperatura_k)) % 2**32)

    # Node features: each spin site
    atom_types = np.full(n_spins, mat_enc, dtype=np.float32)
    local_fields = rng.normal(campo_externo_t, 0.01 * densidad_defectos, n_spins).astype(np.float32)
    resonance_freqs = _base_resonance(material_type) + rng.normal(0, 0.1, n_spins).astype(np.float32)
    spin_values = np.full(n_spins, 0.5 if material_type != "SUPERCONDUCTOR" else 1.0, dtype=np.float32)

    x = np.stack([atom_types, local_fields, resonance_freqs, spin_values], axis=1)

    # Edges: nearest-neighbor chain + some long-range
    src, dst = [], []
    j_vals, distances, int_types = [], [], []

    for i in range(n_spins):
        # Nearest neighbor
        for j in [i - 1, i + 1]:
            if 0 <= j < n_spins:
                src.append(i)
                dst.append(j)
                j_vals.append(j_coupling * (1 + rng.normal(0, 0.05)))
                distances.append(1.0)
                int_types.append(0.0)  # nearest-neighbor

        # Next-nearest with defect-dependent probability
        for j in [i - 2, i + 2]:
            if 0 <= j < n_spins and rng.random() < densidad_defectos:
                src.append(i)
                dst.append(j)
                j_vals.append(j_coupling * 0.1 * (1 + rng.normal(0, 0.1)))
                distances.append(2.0)
                int_types.append(1.0)  # next-nearest

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(
        list(zip(j_vals, distances, int_types)), dtype=torch.float32
    )

    # Global features
    global_features = torch.tensor(
        [[temperatura_k / 300.0, potencia_laser_w * 100.0, campo_externo_t * 1e6]],
        dtype=torch.float32,
    )

    # Target: sensitivity per band (pT/√Hz)
    if target_sensitivity is not None:
        y = torch.tensor([target_sensitivity], dtype=torch.float32)
    else:
        y = None

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        global_features=global_features,
        n_spins=n_spins,
        material_type=material_type,
    )
    return data


def _base_resonance(material_type: str) -> float:
    """Base resonance frequency in GHz for different materials."""
    return {
        "NV_DIAMOND": 2.87,       # NV center zero-field splitting
        "TRAPPED_ION": 12.6,      # Typical Yb+ hyperfine
        "SUPERCONDUCTOR": 5.0,    # Typical transmon
        "OTHER": 1.0,
    }.get(material_type, 1.0)


def generate_synthetic_dataset(n_samples: int = 500) -> DTCGraphDataset:
    """Generate synthetic training data based on physics models.

    Uses theoretical scaling relations to create plausible
    sensitivity values for different DTC configurations.
    """
    dataset = DTCGraphDataset()
    rng = np.random.default_rng(42)

    materials = ["NV_DIAMOND", "TRAPPED_ION", "SUPERCONDUCTOR"]

    for _ in range(n_samples):
        mat = rng.choice(materials)
        n_spins = rng.integers(5, 50)
        temp_k = rng.uniform(0.01, 400.0)
        laser_w = rng.uniform(0.001, 1.0)
        field_t = rng.uniform(0, 1e-6)
        defects = rng.uniform(0, 0.5)

        # Physics-motivated synthetic targets
        base_sens = _synthetic_sensitivity(mat, n_spins, temp_k, laser_w)
        # 3 bands with frequency-dependent scaling
        targets = [
            base_sens * 0.8,   # 0.5-10 Hz (best, low noise)
            base_sens * 1.0,   # 10-50 Hz
            base_sens * 1.5,   # 50-100 Hz (worst, more noise)
        ]
        # Add noise
        targets = [max(0.1, t * (1 + rng.normal(0, 0.1))) for t in targets]

        dataset.add_from_parameters(
            n_spins=n_spins,
            material_type=mat,
            temperatura_k=temp_k,
            potencia_laser_w=laser_w,
            campo_externo_t=field_t,
            densidad_defectos=defects,
            target_sensitivity=targets,
        )

    return dataset


def _synthetic_sensitivity(
    material: str, n_spins: int, temp_k: float, laser_w: float
) -> float:
    """Physics-motivated sensitivity estimate (pT/√Hz).

    Based on scaling: sensitivity ∝ 1/(√N × T2 × γ)
    where T2 depends on temperature and material.
    """
    # Base sensitivity by material (pT/√Hz at optimal conditions)
    base = {"NV_DIAMOND": 200.0, "TRAPPED_ION": 50.0, "SUPERCONDUCTOR": 10.0}.get(material, 500.0)

    # N scaling: sensitivity improves as 1/√N
    n_factor = 1.0 / np.sqrt(max(n_spins, 1))

    # Temperature scaling: worse at higher T (decoherence)
    if material == "SUPERCONDUCTOR":
        # Superconductors need low T
        t_factor = np.exp(temp_k / 5.0) if temp_k > 0.1 else 1.0
    elif material == "TRAPPED_ION":
        # Ions are isolated, less T-dependent in vacuum
        t_factor = 1.0 + 0.001 * temp_k
    else:
        # NV centers: moderate T dependence
        t_factor = 1.0 + 0.01 * temp_k

    # Laser power: more power → better signal (diminishing returns)
    p_factor = 1.0 / (1.0 + np.log1p(laser_w * 10))

    return base * n_factor * t_factor * p_factor

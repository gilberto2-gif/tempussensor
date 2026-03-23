"""Tests for GNN model and dataset."""

import pytest
import torch

from src.ml.gnn_dataset import (
    DTCGraphDataset,
    build_crystal_graph,
    generate_synthetic_dataset,
)
from src.ml.gnn_model import DTCSensorGNN


class TestGNNModel:
    def test_model_creation(self):
        model = DTCSensorGNN()
        assert model.num_output_bands == 3

    def test_forward_pass(self):
        model = DTCSensorGNN()
        data = build_crystal_graph(n_spins=10, material_type="NV_DIAMOND")

        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        preds, uncerts = model(
            data.x, data.edge_index, data.edge_attr,
            batch, data.global_features,
        )

        assert preds.shape == (1, 3)  # 3 frequency bands
        assert uncerts.shape == (1, 3)
        assert (preds > 0).all()  # Softplus ensures positive
        assert (uncerts > 0).all()

    def test_predict_with_confidence(self):
        model = DTCSensorGNN()
        data = build_crystal_graph(n_spins=10, material_type="TRAPPED_ION")

        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        results = model.predict_with_confidence(
            data.x, data.edge_index, data.edge_attr,
            batch, data.global_features,
        )

        assert "0.5-10Hz" in results
        assert "10-50Hz" in results
        assert "50-100Hz" in results
        for band in results.values():
            assert "sensitivity_pT" in band
            assert "uncertainty_pT" in band
            assert "ci_95_lower" in band
            assert "ci_95_upper" in band
            assert band["ci_95_lower"] >= 0


class TestGNNDataset:
    def test_build_crystal_graph(self):
        data = build_crystal_graph(n_spins=10, material_type="NV_DIAMOND")
        assert data.x.shape[0] == 10  # 10 nodes
        assert data.x.shape[1] == 4   # 4 features
        assert data.edge_index.shape[0] == 2
        assert data.edge_attr.shape[1] == 3
        assert data.global_features.shape == (1, 3)

    def test_build_with_target(self):
        data = build_crystal_graph(
            n_spins=5,
            material_type="TRAPPED_ION",
            target_sensitivity=[100.0, 150.0, 200.0],
        )
        assert data.y is not None
        assert data.y.shape == (1, 3)

    def test_dataset_add_from_parameters(self):
        ds = DTCGraphDataset()
        ds.add_from_parameters(
            n_spins=10,
            material_type="NV_DIAMOND",
            temperatura_k=300,
            potencia_laser_w=0.05,
            campo_externo_t=0,
            densidad_defectos=0.1,
            target_sensitivity=[100, 150, 200],
        )
        assert ds.len() == 1
        assert ds.get(0).x.shape[0] == 10

    def test_synthetic_dataset(self):
        ds = generate_synthetic_dataset(n_samples=20)
        assert ds.len() == 20
        for i in range(ds.len()):
            data = ds.get(i)
            assert data.y is not None
            assert data.y.shape == (1, 3)
            assert (data.y > 0).all()


class TestGNNTraining:
    def test_trainer_benchmark(self):
        from src.ml.gnn_train import GNNTrainer

        trainer = GNNTrainer()
        error = trainer._benchmark_source_paper()
        # Error is a percentage — just verify it returns a number
        assert isinstance(error, float)
        assert error >= 0

"""GNN training with early stopping + validation against source paper (confidence 0.52).

Source paper benchmark: ~200 pT/√Hz at 10 Hz in NV diamond.
The model MUST reproduce this within experimental error.
"""

import structlog
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader

from src.ml.gnn_dataset import DTCGraphDataset, build_crystal_graph, generate_synthetic_dataset
from src.ml.gnn_model import DTCSensorGNN

logger = structlog.get_logger(__name__)

# Source paper reference values (confidence 0.52)
SOURCE_PAPER_BENCHMARK = {
    "material": "NV_DIAMOND",
    "n_spins": 15,
    "temperatura_k": 300.0,
    "potencia_laser_w": 0.05,
    "campo_externo_t": 5e-8,  # 50 nT
    "densidad_defectos": 0.1,
    # Expected: ~200 pT/√Hz at 10 Hz
    "expected_sensitivity_10hz": 200.0,
    "tolerance_percent": 30.0,  # ±30% acceptable
}


class GNNTrainer:
    """Trains DTCSensorGNN with early stopping and source paper validation."""

    def __init__(
        self,
        model: DTCSensorGNN | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 20,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or DTCSensorGNN()).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.patience = patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state = None

    def train(
        self,
        train_dataset: DTCGraphDataset | None = None,
        val_dataset: DTCGraphDataset | None = None,
        epochs: int = 200,
        batch_size: int = 32,
    ) -> dict:
        """Train the GNN model.

        If no datasets provided, generates synthetic data.
        Returns training metrics dict.
        """
        if train_dataset is None:
            logger.info("generating_synthetic_data")
            full = generate_synthetic_dataset(n_samples=500)
            # 80/20 split
            n_train = int(0.8 * full.len())
            train_dataset = DTCGraphDataset()
            val_dataset = DTCGraphDataset()
            for i in range(full.len()):
                if i < n_train:
                    train_dataset.add_sample(full.get(i))
                else:
                    val_dataset.add_sample(full.get(i))

        train_loader = DataLoader(
            [train_dataset.get(i) for i in range(train_dataset.len())],
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            [val_dataset.get(i) for i in range(val_dataset.len())],
            batch_size=batch_size,
        )

        loss_fn = nn.HuberLoss(delta=10.0)
        history = {"train_loss": [], "val_loss": [], "benchmark_error": []}

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                preds, _ = self.model(
                    batch.x, batch.edge_index, batch.edge_attr,
                    batch.batch, batch.global_features,
                )
                loss = loss_fn(preds, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train = train_loss / max(n_batches, 1)

            # Validate
            val_loss = self._validate(val_loader, loss_fn)
            self.scheduler.step(val_loss)

            # Benchmark against source paper
            bench_error = self._benchmark_source_paper()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(val_loss)
            history["benchmark_error"].append(bench_error)

            if epoch % 10 == 0:
                logger.info(
                    "training_progress",
                    epoch=epoch,
                    train_loss=round(avg_train, 4),
                    val_loss=round(val_loss, 4),
                    benchmark_error_pct=round(bench_error, 1),
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info("early_stopping", epoch=epoch)
                    break

        # Restore best model
        if self.best_state:
            self.model.load_state_dict(self.best_state)

        final_bench = self._benchmark_source_paper()
        logger.info(
            "training_complete",
            best_val_loss=round(self.best_val_loss, 4),
            final_benchmark_error_pct=round(final_bench, 1),
            source_paper_validated=final_bench <= SOURCE_PAPER_BENCHMARK["tolerance_percent"],
        )

        return {
            "history": history,
            "best_val_loss": self.best_val_loss,
            "benchmark_error_pct": final_bench,
            "source_paper_validated": final_bench <= SOURCE_PAPER_BENCHMARK["tolerance_percent"],
        }

    @torch.no_grad()
    def _validate(self, loader, loss_fn) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(self.device)
            preds, _ = self.model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch.batch, batch.global_features,
            )
            total_loss += loss_fn(preds, batch.y).item()
            n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def _benchmark_source_paper(self) -> float:
        """Validate model against source paper benchmark.

        Returns percentage error vs expected 200 pT/√Hz at 10 Hz.
        """
        self.model.eval()
        b = SOURCE_PAPER_BENCHMARK
        data = build_crystal_graph(
            n_spins=b["n_spins"],
            material_type=b["material"],
            temperatura_k=b["temperatura_k"],
            potencia_laser_w=b["potencia_laser_w"],
            campo_externo_t=b["campo_externo_t"],
            densidad_defectos=b["densidad_defectos"],
            target_sensitivity=[b["expected_sensitivity_10hz"]] * 3,
        ).to(self.device)

        # Single sample — batch index is all zeros
        batch_idx = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        preds, _ = self.model(
            data.x, data.edge_index, data.edge_attr,
            batch_idx, data.global_features,
        )

        # Check 10-50 Hz band (index 1) against expected
        predicted = preds[0, 1].item()
        expected = b["expected_sensitivity_10hz"]
        error_pct = abs(predicted - expected) / expected * 100
        return error_pct

    def save(self, path: str = "models/gnn_dtc_sensor.pt"):
        torch.save(self.model.state_dict(), path)
        logger.info("model_saved", path=path)

    def load(self, path: str = "models/gnn_dtc_sensor.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info("model_loaded", path=path)

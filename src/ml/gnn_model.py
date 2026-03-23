"""GNN model for predicting DTC sensor sensitivity across frequency bands.

Input:  Crystal graph (nodes=atoms/spins, edges=couplings) + global features
Output: Sensitivity (pT/√Hz) for 3 bands: [0.5-10Hz, 10-50Hz, 50-100Hz]
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class DTCSensorGNN(nn.Module):
    """Graph Attention Network for DTC sensor sensitivity prediction.

    Architecture:
    - 3 GATv2Conv layers with residual connections
    - Global features injected after graph pooling
    - 3-head output: one per frequency band
    """

    def __init__(
        self,
        node_features: int = 4,   # [atom_type, local_field, resonance_freq, spin]
        edge_features: int = 3,   # [J_coupling, distance, interaction_type]
        global_features: int = 3, # [temperature_K, laser_power_W, external_field_T]
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_output_bands: int = 3,
    ):
        super().__init__()
        self.num_output_bands = num_output_bands

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # GAT layers with residual connections
        self.conv1 = GATv2Conv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads,
            edge_dim=hidden_dim, dropout=dropout, add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads,
            edge_dim=hidden_dim, dropout=dropout, add_self_loops=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(
            hidden_dim, hidden_dim // num_heads, heads=num_heads,
            edge_dim=hidden_dim, dropout=dropout, add_self_loops=True,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Prediction heads (one per frequency band)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_output_bands),
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_output_bands),
            nn.Softplus(),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Returns:
            (predictions, uncertainties) — both shape [batch_size, 3]
            predictions: sensitivity in pT/√Hz per band
            uncertainties: predicted std dev per band
        """
        # Encode node and edge features
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # GATv2 layers with residual
        h_res = h
        h = self.conv1(h, edge_index, edge_attr=e)
        h = self.norm1(h + h_res)
        h = F.gelu(h)

        h_res = h
        h = self.conv2(h, edge_index, edge_attr=e)
        h = self.norm2(h + h_res)
        h = F.gelu(h)

        h_res = h
        h = self.conv3(h, edge_index, edge_attr=e)
        h = self.norm3(h + h_res)
        h = F.gelu(h)

        # Global mean pooling
        graph_embed = global_mean_pool(h, batch)

        # Encode and concatenate global features
        g = self.global_encoder(global_features)
        combined = torch.cat([graph_embed, g], dim=-1)

        # Predict sensitivity per band + uncertainty
        predictions = self.predictor(combined)
        # Ensure positive sensitivity via softplus
        predictions = F.softplus(predictions)

        uncertainties = self.uncertainty_head(combined)

        return predictions, uncertainties

    def predict_with_confidence(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        global_features: Tensor,
    ) -> dict:
        """Predict sensitivity with confidence intervals."""
        self.eval()
        with torch.no_grad():
            preds, uncerts = self.forward(x, edge_index, edge_attr, batch, global_features)

        band_names = ["0.5-10Hz", "10-50Hz", "50-100Hz"]
        results = {}
        for i, band in enumerate(band_names):
            mean = preds[0, i].item()
            std = uncerts[0, i].item()
            results[band] = {
                "sensitivity_pT": mean,
                "uncertainty_pT": std,
                "ci_95_lower": max(0, mean - 1.96 * std),
                "ci_95_upper": mean + 1.96 * std,
            }
        return results

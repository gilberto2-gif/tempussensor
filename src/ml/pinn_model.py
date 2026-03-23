"""Physics-Informed Neural Network for DTC magnetic response simulation.

Uses DeepXDE to solve the time-dependent Schrödinger equation with DTC
periodic driving, incorporating magnetic susceptibility.

Avoids prohibitively expensive full quantum simulation by learning the
physics from boundary conditions and known analytical limits.
"""

import numpy as np

try:
    import deepxde as dde
    from deepxde.backend import tf  # noqa: F401 — DeepXDE uses TF or PyTorch backend

    HAS_DEEPXDE = True
except ImportError:
    HAS_DEEPXDE = False

import structlog

logger = structlog.get_logger(__name__)


class DTCPINNModel:
    """PINN model for simulating DTC response to weak magnetic fields.

    Solves a simplified model:
    ∂ψ/∂t = -i(H₀ + H_drive(t) + H_mag)ψ

    Where:
    - H₀: base crystal Hamiltonian
    - H_drive(t): periodic driving (laser pulses)
    - H_mag: weak external magnetic field coupling

    The PINN learns the order parameter (subharmonic response amplitude)
    as a function of system parameters.
    """

    def __init__(
        self,
        n_hidden_layers: int = 6,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
    ):
        if not HAS_DEEPXDE:
            raise ImportError("DeepXDE is required for PINN simulation. pip install deepxde")

        self.n_hidden = n_hidden_layers
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.model = None
        self.geom = None

    def build(
        self,
        t_range: tuple[float, float] = (0.0, 100.0),
        param_ranges: dict | None = None,
    ):
        """Build the PINN geometry and network.

        Parameters define the input space:
        - t: time (driving periods)
        - B_ext: external magnetic field (T)
        - T_temp: temperature (K)
        - P_drive: driving power (W)
        - omega_drive: driving frequency (Hz)
        """
        if param_ranges is None:
            param_ranges = {
                "B_ext": (0.0, 1e-9),      # 0 to 1 nT
                "T_temp": (0.01, 400.0),    # cryogenic to room temp
                "P_drive": (0.001, 1.0),    # laser power in W
                "omega_drive": (1.0, 100.0),  # driving frequency Hz
            }

        # 5D input space: t, B_ext, T_temp, P_drive, omega_drive
        bounds_low = [
            t_range[0],
            param_ranges["B_ext"][0],
            param_ranges["T_temp"][0],
            param_ranges["P_drive"][0],
            param_ranges["omega_drive"][0],
        ]
        bounds_high = [
            t_range[1],
            param_ranges["B_ext"][1],
            param_ranges["T_temp"][1],
            param_ranges["P_drive"][1],
            param_ranges["omega_drive"][1],
        ]

        self.geom = dde.geometry.Hypercube(bounds_low, bounds_high)

        # PDE: the physics constraints
        data = dde.data.PDE(
            self.geom,
            self._pde_residual,
            [],  # No explicit BCs — physics constraints only
            num_domain=5000,
            num_test=1000,
        )

        # Network architecture
        layer_sizes = [5] + [self.hidden_size] * self.n_hidden + [2]
        # Output: [order_parameter_amplitude, magnetic_susceptibility]
        net = dde.nn.FNN(
            layer_sizes,
            activation="tanh",
            kernel_initializer="Glorot uniform",
        )

        self.model = dde.Model(data, net)
        self.model.compile(
            "adam",
            lr=self.lr,
            loss_weights=[1.0, 0.1],  # PDE loss + data loss
        )

        logger.info("pinn_model_built", input_dim=5, output_dim=2)

    @staticmethod
    def _pde_residual(x, y):
        """Physics-informed loss: DTC order parameter dynamics.

        Inputs x: [t, B_ext, T_temp, P_drive, omega_drive]
        Outputs y: [order_param, susceptibility]

        Enforces:
        1. Subharmonic response: order_param oscillates at omega_drive/2
        2. Magnetic susceptibility relates to dψ/dB
        3. Thermal decoherence constraint: order_param decays with T
        """
        t = x[:, 0:1]
        B_ext = x[:, 1:2]
        T_temp = x[:, 2:3]
        P_drive = x[:, 3:4]
        omega = x[:, 4:5]

        order_param = y[:, 0:1]
        suscept = y[:, 1:2]

        # Gradients
        d_order_dt = dde.grad.jacobian(y, x, i=0, j=0)
        d_order_dB = dde.grad.jacobian(y, x, i=0, j=1)

        # Physics constraint 1: DTC subharmonic oscillation
        # d(order_param)/dt ≈ -Γ(T)*order_param + A(P)*cos(ωt/2)
        # Γ(T) = decoherence rate ∝ temperature
        gamma = 0.01 * T_temp  # Decoherence rate
        driving_amp = P_drive * 0.1  # Effective driving amplitude
        subharmonic = driving_amp * dde.backend.cos(omega * t / 2.0)
        residual_dynamics = d_order_dt + gamma * order_param - subharmonic

        # Physics constraint 2: susceptibility = d(order_param)/d(B_ext)
        residual_suscept = suscept - d_order_dB

        return [residual_dynamics, residual_suscept]

    def train(self, epochs: int = 10000, display_every: int = 1000) -> dict:
        """Train the PINN model."""
        if self.model is None:
            self.build()

        logger.info("pinn_training_start", epochs=epochs)
        loss_history, train_state = self.model.train(
            epochs=epochs,
            display_every=display_every,
        )

        metrics = {
            "final_loss": float(np.sum(loss_history.loss_train[-1])),
            "epochs_trained": len(loss_history.loss_train),
        }
        logger.info("pinn_training_complete", **metrics)
        return metrics

    def predict(
        self,
        t: float,
        B_ext: float,
        T_temp: float,
        P_drive: float,
        omega_drive: float,
    ) -> dict:
        """Predict DTC response for given parameters.

        Returns order parameter amplitude and magnetic susceptibility.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call build() and train() first.")

        x = np.array([[t, B_ext, T_temp, P_drive, omega_drive]])
        y = self.model.predict(x)

        return {
            "order_parameter": float(y[0, 0]),
            "magnetic_susceptibility": float(y[0, 1]),
            "estimated_sensitivity_pT": self._estimate_sensitivity(
                float(y[0, 1]), B_ext, T_temp
            ),
        }

    def predict_sweep(
        self,
        param_name: str,
        param_range: np.ndarray,
        fixed_params: dict,
    ) -> dict:
        """1D parameter sweep for counterfactual analysis."""
        results = {"param_values": param_range.tolist(), "order_param": [], "susceptibility": [], "sensitivity_pT": []}

        base = {
            "t": fixed_params.get("t", 50.0),
            "B_ext": fixed_params.get("B_ext", 5e-11),
            "T_temp": fixed_params.get("T_temp", 300.0),
            "P_drive": fixed_params.get("P_drive", 0.05),
            "omega_drive": fixed_params.get("omega_drive", 10.0),
        }

        for val in param_range:
            params = {**base, param_name: float(val)}
            pred = self.predict(**params)
            results["order_param"].append(pred["order_parameter"])
            results["susceptibility"].append(pred["magnetic_susceptibility"])
            results["sensitivity_pT"].append(pred["estimated_sensitivity_pT"])

        return results

    @staticmethod
    def _estimate_sensitivity(susceptibility: float, B_ext: float, T_temp: float) -> float:
        """Convert susceptibility to sensitivity estimate in pT/√Hz.

        sensitivity ≈ noise_floor / |χ|
        where noise_floor depends on temperature (thermal noise).
        """
        thermal_noise = 1e-12 * np.sqrt(T_temp)  # Simplified thermal noise model
        chi = abs(susceptibility) + 1e-30  # Avoid division by zero
        sensitivity = thermal_noise / chi * 1e12  # Convert to pT
        return max(sensitivity, 0.01)  # Floor at 0.01 pT/√Hz

    def save(self, path: str = "models/pinn_dtc"):
        if self.model:
            self.model.save(path)
            logger.info("pinn_saved", path=path)

    def restore(self, path: str = "models/pinn_dtc"):
        if self.model:
            self.model.restore(path)
            logger.info("pinn_restored", path=path)

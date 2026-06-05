"""
Neural ODE vs Liquid Neural Network — Complete Python Implementation
====================================================================
Requirements:
    pip install torch torchdiffeq ncps numpy matplotlib

Two self-contained sections:
  1. Neural ODE  — using torchdiffeq for the ODE solver
  2. Liquid NN   — using the ncps library (LTC + CfC wiring)
  3. Side-by-side comparison on a sine-wave prediction task
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Neural ODE solver
from torchdiffeq import odeint, odeint_adjoint

# Liquid Neural Networks (Neural Circuit Policies)
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP


# ──────────────────────────────────────────────────────────────────────────────
# 1. NEURAL ODE
# ──────────────────────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    The right-hand side of  dh/dt = f(h, t, θ).
    This small network IS the derivative of the hidden state.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim),
        )
        # Track how many NFEs (number of function evaluations) the solver uses
        self.nfe = 0

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.net(h)


class NeuralODE(nn.Module):
    """
    Full Neural ODE model:
      x  →  encoder  →  h₀  →  ODE solve  →  h(T)  →  decoder  →  ŷ
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 use_adjoint: bool = True):
        super().__init__()
        self.encoder  = nn.Linear(input_dim, hidden_dim)
        self.odefunc  = ODEFunc(hidden_dim)
        self.decoder  = nn.Linear(hidden_dim, output_dim)
        # adjoint saves memory; regular odeint stores full trajectory
        self.solver   = odeint_adjoint if use_adjoint else odeint

    def forward(self, x: torch.Tensor,
                t_span: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:      (batch, input_dim)
            t_span: 1-D tensor of time points, e.g. torch.linspace(0, 1, 10)
        Returns:
            ŷ:  (batch, output_dim)  — prediction at final time t[-1]
        """
        if t_span is None:
            t_span = torch.linspace(0.0, 1.0, 10)

        h0 = self.encoder(x)                          # (batch, hidden)
        # odeint returns (len(t_span), batch, hidden)
        trajectory = self.solver(
            self.odefunc,
            h0,
            t_span,
            method="dopri5",                          # adaptive Runge-Kutta
            rtol=1e-3,
            atol=1e-4,
        )
        h_final = trajectory[-1]                      # take h at last time step
        return self.decoder(h_final)

    @property
    def nfe(self) -> int:
        return self.odefunc.nfe

    def reset_nfe(self):
        self.odefunc.nfe = 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. LIQUID NEURAL NETWORK  (two flavours: LTC and CfC)
# ──────────────────────────────────────────────────────────────────────────────

class LiquidNeuralNetwork(nn.Module):
    """
    Wraps ncps.torch.LTC or CfC in a simple encoder → liquid → decoder pipeline.

    LTC  = Liquid Time-Constant cell   (ODE-based, requires a solver internally)
    CfC  = Closed-form Continuous-time (no solver, fastest, default choice)

    The wiring uses AutoNCP which automatically builds a sparse
    inter-neuron connectivity resembling biological neural circuits.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 cell_type: str = "cfc"):
        super().__init__()

        # AutoNCP builds a biologically-inspired wiring with:
        #   - sensory (input) neurons
        #   - inter neurons
        #   - command neurons
        #   - motor (output) neurons
        wiring = AutoNCP(
            units=hidden_dim,
            output_size=output_dim,
            sparsity_level=0.5,       # 50% sparse connections
        )

        if cell_type.lower() == "ltc":
            self.rnn = LTC(input_dim, wiring, batch_first=True)
        else:  # default: CfC (closed-form, ~10× faster than LTC)
            self.rnn = CfC(input_dim, wiring, batch_first=True)

        self.cell_type = cell_type

    def forward(self, x: torch.Tensor,
                timespans: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:          (batch, seq_len, input_dim)
            timespans:  (batch, seq_len) — elapsed time Δt between steps.
                        If None, assumes uniform Δt = 1.
        Returns:
            output:     (batch, seq_len, output_dim)
        """
        if timespans is not None:
            # ncps accepts time_delta as keyword argument
            output, _ = self.rnn(x, timespans=timespans)
        else:
            output, _ = self.rnn(x)
        return output


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA: NOISY SINE WAVE  (shared toy task)
# ──────────────────────────────────────────────────────────────────────────────

def make_sine_data(n_samples: int = 512, seq_len: int = 20,
                   noise: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        X:  (n_samples, seq_len, 1)   — noisy sine inputs
        Y:  (n_samples, seq_len, 1)   — clean sine targets (1-step ahead)
    """
    t = torch.linspace(0, 4 * np.pi, seq_len + 1).unsqueeze(0)  # (1, seq+1)
    t = t.expand(n_samples, -1)

    # Random phase shift per sample
    phase = torch.rand(n_samples, 1) * 2 * np.pi
    sine  = torch.sin(t + phase)

    X = sine[:, :-1, None] + noise * torch.randn(n_samples, seq_len, 1)
    Y = sine[:, 1:,  None]
    return X, Y


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def train_neural_ode(epochs: int = 30, lr: float = 1e-3,
                     hidden: int = 32) -> tuple[NeuralODE, list]:
    """
    Train Neural ODE on a point-prediction task:
    flatten the time series → predict next value.
    """
    X, Y = make_sine_data()
    # Use only the last value in each sequence as input / target for simplicity
    x_in  = X[:, -1, :]          # (batch, 1)
    y_tgt = Y[:, -1, :]          # (batch, 1)

    model = NeuralODE(input_dim=1, hidden_dim=hidden, output_dim=1)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        model.reset_nfe()
        pred = model(x_in)
        loss = loss_fn(pred, y_tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"[NeuralODE]  epoch {epoch+1:3d} | loss {loss.item():.5f} "
                  f"| NFE {model.nfe}")
    return model, losses


def train_liquid_nn(epochs: int = 30, lr: float = 1e-3,
                    hidden: int = 32,
                    cell_type: str = "cfc") -> tuple[LiquidNeuralNetwork, list]:
    """
    Train Liquid NN on the full sequence prediction task (seq2seq).
    """
    X, Y = make_sine_data()

    model   = LiquidNeuralNetwork(input_dim=1, hidden_dim=hidden,
                                  output_dim=1, cell_type=cell_type)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        pred = model(X)              # (batch, seq, 1)
        loss = loss_fn(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"[LiquidNN/{cell_type}] epoch {epoch+1:3d} | loss {loss.item():.5f}")
    return model, losses


# ──────────────────────────────────────────────────────────────────────────────
# 5. DEMONSTRATION: IRREGULAR TIME STEPS  (Liquid NN advantage)
# ──────────────────────────────────────────────────────────────────────────────

def demo_irregular_timestamps():
    """
    Show how LiquidNN handles non-uniform time gaps natively.
    This is one of the key practical advantages over vanilla RNNs.
    """
    print("\n── Irregular timestamps demo ──")
    batch, seq = 4, 10
    x = torch.randn(batch, seq, 1)

    # Simulate irregular Δt (e.g. sensor data with variable polling rates).
    # ncps CfC expects timespans shaped (1, seq): the library slices timespans[:, t]
    # and calls .squeeze(), so the per-step value must be a scalar (requires batch
    # dim == 1). Per-sample Δt is not supported by the library's current design.
    dt = torch.abs(torch.randn(1, seq)) * 0.5 + 0.1   # (1, seq)

    model = LiquidNeuralNetwork(input_dim=1, hidden_dim=16, output_dim=1)
    model.eval()
    with torch.no_grad():
        out = model(x, timespans=dt)
    print(f"Input shape:      {x.shape}")
    print(f"Time-delta shape: {dt.shape}  (non-uniform Δt per step, shared across batch)")
    print(f"Output shape:     {out.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(losses_node: list, losses_liquid: list,
                    filename: str = "training_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Neural ODE vs Liquid NN — Training Loss", fontsize=13)

    axes[0].plot(losses_node, color="#534AB7", linewidth=2)
    axes[0].set_title("Neural ODE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(losses_liquid, color="#D85A30", linewidth=2)
    axes[1].set_title("Liquid NN (CfC)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE Loss")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"\nPlot saved → {filename}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Neural ODE training")
    print("=" * 60)
    node_model, node_losses = train_neural_ode(epochs=30)

    print("\n" + "=" * 60)
    print("  Liquid Neural Network (CfC) training")
    print("=" * 60)
    liquid_model, liquid_losses = train_liquid_nn(epochs=30, cell_type="cfc")

    demo_irregular_timestamps()

    plot_comparison(node_losses, liquid_losses,
                    filename="training_comparison.png")

    # ── Quick model summary ──
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("\n── Parameter counts ──")
    print(f"  Neural ODE  : {count_params(node_model):,} params")
    print(f"  Liquid NN   : {count_params(liquid_model):,} params")

    print("\nDone! ✓")
import time
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch_optimizer as optim_extra
import PINNmodel

# -----------------------------
# Fixed experimental conditions
# -----------------------------
pH = 12.5
ch = 1.0
k = 103.4
T = 298.15                  # K
Pno = 101.325                # kPa
Hno = 709.1725
cno = Pno / Hno              # mol/L
ooh = ch / (1.0 + 10 ** (11.73 - pH))  # mol/L (bulk HOO-)

# RPB geometry
rpm = 1200.0
r1 = 26e-3                   # m
r2 = 70e-3                   # m
r = np.sqrt((r1**2 + r2**2) / 2.0)  # m
h = 25e-3                    # m

# Flowrates
Q = 20e-3 / 3600.0           # m^3/s
gn2 = 2000e-3 / 3600.0       # m^3/s

# -----------------------------
# De vs rpm (interpolation, used for quick reporting only)
# -----------------------------
rpm_data = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600], dtype=float)
De_data = np.array([5.3064, 6.2257, 6.7727, 7.1618, 7.4629, 7.7081, 7.9144, 8.0818], dtype=float)
f_De = interp1d(rpm_data, De_data, kind="linear")
def get_de(rpm_value: float) -> float:
    """Return interpolated De for a given rpm (no extrapolation)."""
    if rpm_value < rpm_data.min() or rpm_value > rpm_data.max():
        raise ValueError("rpm is out of the interpolation range (extrapolation/data needed).")
    return float(f_De(rpm_value))

# -----------------------------
# Training configuration
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PINNmodel.setup_seed(42)

epochs = 25000
N = 15000
N1 = 5000
nt_eta = 100

USE_ETA = True
USE_CFD = False
CFD_FILE = "0.0012736-r.csv"
ETA_FILE = "data.xlsx"
T_FIX = 0.0012736  # fixed physical time for CFD snapshot (if used)

# -----------------------------
# Load experimental eta data
# -----------------------------
exp_data = PINNmodel.load_exp_data(ETA_FILE) if USE_ETA else []

# -----------------------------
# Optional: load CFD radial profiles
# -----------------------------
if USE_CFD:
    df = pd.read_csv(CFD_FILE, header=None)
    x_fem_NO = df[0].values
    c1 = torch.tensor(df[1].values, dtype=torch.float32, device=device)
    x_fem_OOH = df[2].dropna().values
    c2 = torch.tensor(df[3].dropna().values, dtype=torch.float32, device=device)
else:
    x_fem_NO, x_fem_OOH, c1, c2 = None, None, None, None

# -----------------------------
# Build PINN model
# -----------------------------
model = PINNmodel.CoupledPINN(
    cNO_star=cno,
    cOOH_star=ooh,
    n_hidden_cNO=128,
    n_hidden_cOOH=128,
    n_hidden_shared=128,
    activation_cNO=nn.Tanh,
    activation_cOOH=nn.Tanh,
    activation_shared=nn.Tanh,
    n_layers_cNO=7,
    n_layers_cOOH=7,
    n_layers_shared=6,
    dropout_prob_cNO=0.0,
    dropout_prob_cOOH=0.0,
).to(device)

(
    X_eq, X_ic, X_bc_left, X_bc_right,
    X_fem_NO, X_fem_OOH,
    c_fem_NO_tensor, c_fem_OOH_tensor
) = PINNmodel.generate_sampling_points(
    N=N,
    N1=N1,
    pH_range=(10.0, 14.0),
    ch_range=(0.01, 1.6),
    rpm_range=(200.0, 1600.0),
    gn2=gn2,
    x_range=(0.0, 1.0),
    t_range=(0.0, 1.0),
    x_fem_NO=x_fem_NO,
    x_fem_OOH=x_fem_OOH,
    c_fem_NO=c1,
    c_fem_OOH=c2,
    t_fix=T_FIX,
    device=device,
    sampling_method="sobol",
)

# -----------------------------
# Logs / checkpoints
# -----------------------------
save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

losses = []
losses_cNO = []
losses_cOOH = []
losses_pde = []
losses_ic = []
losses_phys = []
losses_eta = []
lrs = []

# -----------------------------
# Optimizer: Adam + Lookahead
# -----------------------------
start_time = time.time()
base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
optimizer = optim_extra.Lookahead(base_optimizer, k=5, alpha=0.5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=100)

# EMA warmup for stable initial loss ratios (GradNorm)
class EMA:
    def __init__(self, alpha=0.1, eps=1e-8):
        self.alpha = alpha
        self.ema = None
        self.eps = eps
    def update(self, new_value: float) -> float:
        if self.ema is None:
            self.ema = float(new_value) + self.eps
        else:
            self.ema = self.alpha * float(new_value) + (1 - self.alpha) * self.ema
        return self.ema
ema_trackers = {key: EMA(alpha=0.1) for key in ["cNO", "cOOH"]}
initial_ema_steps = 100
initial_losses = None

for epoch in range(epochs + initial_ema_steps):
    (
        L_pde, L_ic, L_bc,
        L_pde_cNO, L_pde_cOOH,
        L_ic_cNO, L_ic_cOOH,
        L_bc_left_cNO, L_bc_left_cOOH,
        L_bc_right_cNO, L_bc_right_cOOH,
        L_cNO, L_cOOH,
        L_fem, L_etaNO, L_phys
    ) = PINNmodel.multi_task_loss(
        model,
        X_eq, X_ic, X_bc_left, X_bc_right,
        X_fem_NO, X_fem_OOH,
        cno=cno,
        r=r, r2=r2, r1=r1,
        u1=c_fem_NO_tensor, u2=c_fem_OOH_tensor,
        Hno=Hno, Pno=Pno, T=T, h=h,
        nt_eta=nt_eta,
        exp_data=exp_data,
        k=k,
        use_cfd=USE_CFD,
        use_eta=USE_ETA,
        device=device,
    )

    task_losses = torch.stack([L_cNO, L_cOOH])  # two tasks for GradNorm
    losses_eta.append(float(L_etaNO.item()))

    # EMA warmup phase
    if epoch < initial_ema_steps:
        ema_trackers["cNO"].update(L_cNO.item())
        ema_trackers["cOOH"].update(L_cOOH.item())

        weights = model.get_weights()
        weighted_loss = (weights * task_losses).sum()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        scheduler.step(weighted_loss.item())

        if epoch % 10 == 0:
            gnorm = PINNmodel.print_grad_norm(model)
            print(
                f"[EMA {epoch:04d}] "
                f"L_cNO={ema_trackers['cNO'].ema:.4e}, "
                f"L_cOOH={ema_trackers['cOOH'].ema:.4e}, "
                f"grad_norm={gnorm:.4e}"
            )

        if epoch == initial_ema_steps - 1:
            initial_losses = np.array([ema_trackers["cNO"].ema, ema_trackers["cOOH"].ema], dtype=float)
            print("Initial losses (EMA):", initial_losses)

        continue

    # Redundant guard
    if epoch == initial_ema_steps and initial_losses is None:
        initial_losses = np.array([ema_trackers["cNO"].ema, ema_trackers["cOOH"].ema], dtype=float)
        print(f"Initial losses set at epoch {epoch}: {initial_losses}")

    # Main training step
    optimizer.zero_grad()

    weights = model.get_weights()
    weighted_loss = (weights * task_losses).sum()
    weighted_loss.backward(retain_graph=True)

    losses.append(float(weighted_loss.item()))
    losses_cNO.append(float(L_cNO.item()))
    losses_cOOH.append(float(L_cOOH.item()))
    losses_pde.append(float(L_pde.item()))
    losses_ic.append(float(L_ic.item()))
    losses_phys.append(float(L_phys.item()))

    # GradNorm update (weights gradient)
    if model.weights.grad is not None:
        model.weights.grad.data.zero_()

    grad_norm_loss = PINNmodel.gradnorm_update(model, task_losses, initial_losses, alpha=1.0)

    pre_clip_norm = PINNmodel.print_grad_norm(model)
    clip_params = PINNmodel.get_clip_params(model)
    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=3e2)

    optimizer.step()
    scheduler.step(weighted_loss.item())
    lrs.append(float(optimizer.param_groups[0]["lr"]))

    # Normalize weights so that sum(weights) == number_of_tasks (2)
    with torch.no_grad():
        w = model.get_weights()
        w = w * (2.0 / (w.sum() + 1e-8))
        model.weights.data = torch.log(torch.exp(w) - 1.0 + 1e-8)

    if epoch % 50 == 0:
        post_clip_norm = PINNmodel.print_grad_norm(model)
        print(
            f"[Epoch {epoch:06d}] "
            f"L_cNO={L_cNO.item():.4e}, L_cOOH={L_cOOH.item():.4e}, "
            f"L_eta={L_etaNO.item():.4e}, L_phys={L_phys.item():.4e}, "
            f"L_pde={L_pde.item():.4e}, L_ic={L_ic.item():.4e}, "
            f"L_bc={L_bc.item():.4e}, L_fem={L_fem.item():.4e}, "
            f"loss={weighted_loss.item():.4e}, gradnorm={grad_norm_loss.item():.4e}, "
            f"w={model.get_weights().detach().cpu().numpy()}, lr={optimizer.param_groups[0]['lr']:.3e}, "
            f"grad_pre={pre_clip_norm:.4e}, grad_post={post_clip_norm:.4e}"
        )

    if epoch % 2000 == 0:
        ckpt_path = save_dir / f"coupled_pinn_la_epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": losses,
                "losses_cNO": losses_cNO,
                "losses_cOOH": losses_cOOH,
                "losses_pde": losses_pde,
                "losses_ic": losses_ic,
                "losses_phys": losses_phys,
                "losses_eta": losses_eta,
                "lrs": lrs,
                "initial_losses": initial_losses,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

# -----------------------------
# Final save
# -----------------------------
end_time = time.time()
duration_seconds = end_time - start_time
print(f"Training finished at: {datetime.datetime.now()}")
print(f"Total time: {duration_seconds:.4f} s ({datetime.timedelta(seconds=duration_seconds)})")
torch.save(
    {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "losses": losses,
        "losses_cNO": losses_cNO,
        "losses_cOOH": losses_cOOH,
        "losses_pde": losses_pde,
        "losses_ic": losses_ic,
        "losses_phys": losses_phys,
        "losses_eta": losses_eta,
        "duration_seconds": duration_seconds,
        "lrs": lrs,
    },
    "final_eta.pth",
)

import numpy as np
from numpy import pi, exp, log, sqrt, diff
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
import random
from math import pi, log
import pandas as pd

# Reproducibility
def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

# Autograd helpers
def gradients(u, x, order: int = 1):
    if order == 1:
        return torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            only_inputs=True
        )[0]
    return gradients(gradients(u, x), x, order=order - 1)

def partial_derivative_n(u, X, idx: int, order: int = 1):
    """Compute the n-th partial derivative of u(X) w.r.t. X[:, idx]."""
    grad = u
    for _ in range(order):
        grad = torch.autograd.grad(
            grad, X,
            grad_outputs=torch.ones_like(grad),
            create_graph=True,
            retain_graph=True
        )[0][:, idx:idx + 1]
    return grad

# Property/geometry utilities
def get_d(rpm, Q, r):
    """Droplet diameter d (supports tensor broadcasting)."""
    omega = 2 * pi * rpm / 60.0
    d = 12.84 * ((72.0e-2) / (r * omega ** 2 * 1000.0)) ** 0.630 * ((Q * 3600) ** 0.201)
    return d
def get_ts(rpm, Q, r, r2, r1):
    """Characteristic time scale ts."""
    omega = 2 * pi * rpm / 60.0
    u0 = 0.02107 * Q ** 0.2279 * (omega ** 2 * r)
    ts = (r2 - r1) / (u0 * 31.0)
    return ts
def get_holdup(rpm, Q, r, v0=1.0e-6, g0=100):
    """Holdup (dimensionless)."""
    omega = 2 * pi * rpm / 60.0
    u0 = 0.02107 * Q ** 0.2279 * (omega ** 2 * r)
    holdup = 0.039 * ((r * omega ** 2 / g0) ** (-0.5)) * ((u0 / 0.01) ** 0.6) * ((1.35e-6 / v0) ** 0.22)
    return holdup

# De lookup (rpm -> De) via interpolation
rpm_data = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
De_data = np.array([5.3064, 6.2257, 6.7727, 7.1618, 7.4629, 7.7081, 7.9144, 8.0818])
f_De = interp1d(rpm_data, De_data, kind="linear", fill_value="extrapolate")
def get_de(rpm):
    """Interpolate De for a given rpm (tensor or scalar)."""
    if torch.is_tensor(rpm):
        rpm_np = rpm.detach().cpu().numpy()
        De_np = f_De(rpm_np)
        De = torch.tensor(De_np, dtype=rpm.dtype, device=rpm.device)
        if rpm.requires_grad:
            De.requires_grad_()
        return De
    return f_De(rpm)

# Networks
class SharedLayer(nn.Module):
    def __init__(self, n_input=6, n_hidden=48, n_layers=4, activation=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden)]
        nn.init.xavier_normal_(layers[-1].weight)
        layers.append(activation())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers.append(activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class cNOPINN(nn.Module):
    def __init__(self, n_hidden=32, n_output=1, n_layers=2, activation=nn.Tanh, dropout_prob=0.2):
        super().__init__()
        layers = [nn.Linear(n_hidden, n_hidden)]
        nn.init.xavier_normal_(layers[-1].weight)
        layers += [activation(), nn.Dropout(p=dropout_prob)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers += [activation(), nn.Dropout(p=dropout_prob)]
        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class cOOHPINN(nn.Module):
    def __init__(self, n_hidden=32, n_output=1, n_layers=2, activation=nn.Tanh, dropout_prob=0.2):
        super().__init__()
        layers = [nn.Linear(n_hidden, n_hidden)]
        nn.init.xavier_normal_(layers[-1].weight)
        layers += [activation(), nn.Dropout(p=dropout_prob)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers += [activation(), nn.Dropout(p=dropout_prob)]
        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class CoupledPINN(nn.Module):
    """Shared trunk + two task heads (cNO, cOOH) with trainable task weights."""
    def __init__(
        self,
        cNO_star,
        cOOH_star,
        n_hidden_shared=48,
        n_hidden_cNO=40,
        n_hidden_cOOH=40,
        n_layers_shared=6,
        n_layers_cNO=4,
        n_layers_cOOH=4,
        activation_shared=nn.Tanh,
        activation_cNO=nn.Tanh,
        activation_cOOH=nn.Tanh,
        dropout_prob_cNO=0.2,
        dropout_prob_cOOH=0.2,
    ):
        super().__init__()
        self.cNO_star = cNO_star
        self.cOOH_star = cOOH_star

        self.shared = SharedLayer(
            n_input=6,
            n_hidden=n_hidden_shared,
            n_layers=n_layers_shared,
            activation=activation_shared,
        )
        self.nn_cNO = cNOPINN(
            n_hidden=n_hidden_cNO,
            n_output=1,
            n_layers=n_layers_cNO,
            activation=activation_cNO,
            dropout_prob=dropout_prob_cNO,
        )
        self.nn_cOOH = cOOHPINN(
            n_hidden=n_hidden_cOOH,
            n_output=1,
            n_layers=n_layers_cOOH,
            activation=activation_cOOH,
            dropout_prob=dropout_prob_cOOH,
        )
        # Trainable task weights (GradNorm-style); constrained via softplus in get_weights()
        self.weights = nn.Parameter(0.5413 * torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        x_ = x[:, 0:1]
        t_ = x[:, 1:2]
        pH = x[:, 2:3]
        ch = x[:, 3:4]
        cOOH_star_dynamic = ch / (1.0 + 10 ** (11.73 - pH))

        shared_out = self.shared(x)
        cNO_hat = self.nn_cNO(shared_out)
        cOOH_hat = self.nn_cOOH(shared_out)

        # Enforce boundary-friendly parametrization
        cNO = (x_ - 1) ** 2 * (cNO_hat * x_ + (self.cNO_star) ** 0.5) ** 2
        cOOH = (x_) ** 2 * (cOOH_hat * (1 - x_) + (cOOH_star_dynamic) ** 0.5) ** 2
        return cNO, cOOH

    def get_shared_layer(self):
        return list(self.shared.parameters())

    def get_weights(self):
        return torch.nn.functional.softplus(self.weights)

class SingleNetworkPINN(nn.Module):
    """Single network with 2 outputs (cNO, cOOH), using the same boundary parametrization."""

    def __init__(
        self,
        cNO_star,
        cOOH_star,
        n_input=6,
        n_hidden=48,
        n_layers=8,
        n_output=2,
        activation=nn.Tanh,
        dropout_prob=0.2,
    ):
        super().__init__()
        self.cNO_star = cNO_star
        self.cOOH_star = cOOH_star

        layers = [nn.Linear(n_input, n_hidden)]
        nn.init.xavier_normal_(layers[-1].weight)
        layers.append(activation())
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers.append(activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_ = x[:, 0:1]
        t_ = x[:, 1:2]
        pH = x[:, 2:3]
        ch = x[:, 3:4]
        cOOH_star_dynamic = ch / (1.0 + 10 ** (11.73 - pH))

        raw_outputs = self.net(x)
        cNO_hat = raw_outputs[:, 0:1]
        cOOH_hat = raw_outputs[:, 1:2]

        cNO = (x_ - 1) ** 2 * (cNO_hat * x_ + (self.cNO_star) ** 0.5) ** 2
        cOOH = (x_) ** 2 * (cOOH_hat * (1 - x_) + (cOOH_star_dynamic) ** 0.5) ** 2
        return cNO, cOOH


class SingleTaskNet(nn.Module):
    """Reusable single-task MLP head."""

    def __init__(
        self,
        n_input=6,
        n_hidden=128,
        n_layers=13,
        n_output=1,
        activation=nn.Tanh,
        dropout_prob=0.0,
    ):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden)]
        nn.init.xavier_normal_(layers[-1].weight)
        layers.append(activation())
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers.append(activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class IndependentPINN(nn.Module):
    """Two fully independent networks for cNO and cOOH."""

    def __init__(
        self,
        cNO_star,
        cOOH_star,
        n_hidden_cNO=128,
        n_layers_cNO=13,
        activation_cNO=nn.Tanh,
        dropout_prob_cNO=0.0,
        n_hidden_cOOH=128,
        n_layers_cOOH=13,
        activation_cOOH=nn.Tanh,
        dropout_prob_cOOH=0.0,
    ):
        super().__init__()
        self.cNO_star = cNO_star
        self.cOOH_star = cOOH_star

        self.net_cNO = SingleTaskNet(
            n_input=6,
            n_hidden=n_hidden_cNO,
            n_layers=n_layers_cNO,
            n_output=1,
            activation=activation_cNO,
            dropout_prob=dropout_prob_cNO,
        )
        self.net_cOOH = SingleTaskNet(
            n_input=6,
            n_hidden=n_hidden_cOOH,
            n_layers=n_layers_cOOH,
            n_output=1,
            activation=activation_cOOH,
            dropout_prob=dropout_prob_cOOH,
        )

    def forward(self, x):
        x_ = x[:, 0:1]
        t_ = x[:, 1:2]
        pH = x[:, 2:3]
        ch = x[:, 3:4]
        cOOH_star_dynamic = ch / (1.0 + 10 ** (11.73 - pH))

        cNO_hat = self.net_cNO(x)
        cOOH_hat = self.net_cOOH(x)

        cNO = (x_ - 1) ** 2 * (cNO_hat * x_ + (self.cNO_star) ** 0.5) ** 2
        cOOH = (x_) ** 2 * (cOOH_hat * (1 - x_) + (cOOH_star_dynamic) ** 0.5) ** 2
        return cNO, cOOH


def print_grad_norm(model):
    """Compute global grad norm excluding task-weight params."""
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None and "weights" not in name:
            param_norm = param.grad.norm(2).item()
            total_norm += param_norm ** 2
    return total_norm ** 0.5


def get_clip_params(model):
    """Parameters eligible for gradient clipping (excluding task weights)."""
    return [p for n, p in model.named_parameters() if p.grad is not None and "weights" not in n]

# CFD targets (precomputed)
u1, u2 = [torch.tensor(np.loadtxt(f"u{i}x200.csv", delimiter=","), dtype=torch.float32) for i in (1, 2)]

def generate_sampling_points(N=20000,N1=5000,pH_range=(10.0, 13.0),ch_range=(0.05, 1.6),rpm_range=(200.0, 1600.0),gn2=2.0 / 3600,x_range=(0.0, 1.0),t_range=(0.0, 1.0),x_fem_NO=None,x_fem_OOH=None,c_fem_NO=None,c_fem_OOH=None,t_fix=1.0,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),sampling_method="sobol",):
    """
    Generate sampling points for multi-condition PINN.
    Input dimension: [x*, t*, pH, c_h, rpm, Q].
    x_range/t_range can be (min,max) or a list of intervals [(a,b), ...].
    """

    def generate_samples(dim, num_samples, method):
        if num_samples <= 0:
            return torch.empty(0, dim)
        if method == "random":
            return torch.rand(num_samples, dim)
        if method == "sobol":
            try:
                soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
                return soboleng.draw(num_samples)
            except Exception as e:
                print(f"Warning: PyTorch SobolEngine failed: {e}")
                try:
                    import sobol_seq
                    return torch.tensor(sobol_seq.i4_sobol_generate(dim, num_samples), dtype=torch.float32)
                except ImportError:
                    print("Warning: sobol_seq not available; fallback to random sampling.")
                    return torch.rand(num_samples, dim)
        print(f"Warning: unsupported sampling method '{method}', using random sampling.")
        return torch.rand(num_samples, dim)

    def _sample_from_intervals(num_samples, intervals, method):
        """Sample from one or multiple intervals."""
        if num_samples <= 0:
            return torch.empty(0, 1)

        lengths = [end - start for start, end in intervals]
        total_length = sum(lengths)
        if total_length <= 1e-9:
            return torch.full((num_samples, 1), intervals[0][0])

        samples_per_interval = [int(np.round(num_samples * length / total_length)) for length in lengths]
        diff_ = num_samples - sum(samples_per_interval)
        if diff_ != 0:
            idx_max_len = int(np.argmax(lengths))
            samples_per_interval[idx_max_len] += diff_

        all_samples = []
        for i, (start, end) in enumerate(intervals):
            n_sub = samples_per_interval[i]
            if n_sub > 0:
                p = generate_samples(1, n_sub, method)
                all_samples.append(start + (end - start) * p)

        if not all_samples:
            return torch.empty(0, 1)

        result = torch.cat(all_samples, dim=0)
        return result[torch.randperm(result.shape[0])]

    if isinstance(x_range, tuple):
        x_range = [x_range]
    if isinstance(t_range, tuple):
        t_range = [t_range]

    Q_range = (gn2 / 120, gn2 / 30)

    # PDE interior points
    p = generate_samples(4, N, sampling_method)
    pH_vals = pH_range[0] + (pH_range[1] - pH_range[0]) * p[:, 0:1]
    ch_vals = ch_range[0] + (ch_range[1] - ch_range[0]) * p[:, 1:2]
    rpm_vals = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p[:, 2:3]
    Q_vals = Q_range[0] + (Q_range[1] - Q_range[0]) * p[:, 3:4]

    x_vals = _sample_from_intervals(N, x_range, sampling_method)
    t_vals = _sample_from_intervals(N, t_range, sampling_method)
    X_eq = torch.cat([x_vals, t_vals, pH_vals, ch_vals, rpm_vals, Q_vals], dim=1)

    # Initial condition points (t = t_min)
    num_ic_samples = N1 // 3
    p_ic = generate_samples(4, num_ic_samples, sampling_method)
    pH_ic = pH_range[0] + (pH_range[1] - pH_range[0]) * p_ic[:, 0:1]
    ch_ic = ch_range[0] + (ch_range[1] - ch_range[0]) * p_ic[:, 1:2]
    rpm_ic = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p_ic[:, 2:3]
    Q_ic = Q_range[0] + (Q_range[1] - Q_range[0]) * p_ic[:, 3:4]

    x_ic = _sample_from_intervals(num_ic_samples, x_range, sampling_method)
    t_ic = torch.full_like(x_ic, min(start for start, end in t_range))
    X_ic = torch.cat([x_ic, t_ic, pH_ic, ch_ic, rpm_ic, Q_ic], dim=1)

    # Boundary points
    X_bc_left, X_bc_right = None, None
    num_bc_samples = N1 // 3
    global_min_x = min(start for start, end in x_range)
    global_max_x = max(end for start, end in x_range)

    if global_min_x <= 1e-9:
        p_left = generate_samples(4, num_bc_samples, sampling_method)
        pH_left = pH_range[0] + (pH_range[1] - pH_range[0]) * p_left[:, 0:1]
        ch_left = ch_range[0] + (ch_range[1] - ch_range[0]) * p_left[:, 1:2]
        rpm_left = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p_left[:, 2:3]
        Q_left = Q_range[0] + (Q_range[1] - Q_range[0]) * p_left[:, 3:4]

        t_left = _sample_from_intervals(num_bc_samples, t_range, sampling_method)
        x_left = torch.zeros_like(t_left)
        X_bc_left = torch.cat([x_left, t_left, pH_left, ch_left, rpm_left, Q_left], dim=1).to(device)

    if global_max_x >= 1.0 - 1e-9:
        p_right = generate_samples(4, num_bc_samples, sampling_method)
        pH_right = pH_range[0] + (pH_range[1] - pH_range[0]) * p_right[:, 0:1]
        ch_right = ch_range[0] + (ch_range[1] - ch_range[0]) * p_right[:, 1:2]
        rpm_right = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p_right[:, 2:3]
        Q_right = Q_range[0] + (Q_range[1] - Q_range[0]) * p_right[:, 3:4]

        t_right = _sample_from_intervals(num_bc_samples, t_range, sampling_method)
        x_right = torch.ones_like(t_right)
        X_bc_right = torch.cat([x_right, t_right, pH_right, ch_right, rpm_right, Q_right], dim=1).to(device)

    # Optional CFD data points (normalized by d/2 and ts)
    X_fem_NO, X_fem_OOH = None, None
    c_fem_NO_tensor, c_fem_OOH_tensor = None, None

    if (
        x_fem_NO is not None and c_fem_NO is not None and
        x_fem_OOH is not None and c_fem_OOH is not None
    ):
        d_fem = get_d(rpm=1200, Q=20e-3 / 3600, r=0.052802)
        ts_fem = get_ts(rpm=1200, Q=20e-3 / 3600, r=0.052802, r2=0.07, r1=0.026)

        def filter_and_create_fem_tensors(x_fem_raw, c_fem_raw, intervals):
            x_arr = x_fem_raw.cpu().numpy() if torch.is_tensor(x_fem_raw) else np.array(x_fem_raw)
            x_norm = x_arr / (d_fem / 2)

            mask = np.zeros_like(x_norm, dtype=bool)
            for start, end in intervals:
                mask |= (x_norm >= start) & (x_norm <= end)

            if not np.any(mask):
                return None, None

            x_filtered = x_norm[mask]
            c_filtered = (c_fem_raw.cpu().numpy() if torch.is_tensor(c_fem_raw) else np.array(c_fem_raw))[mask]

            x_tensor = torch.tensor(x_filtered, dtype=torch.float32).reshape(-1, 1)
            c_tensor = torch.tensor(c_filtered, dtype=torch.float32).reshape(-1, 1)

            npts = x_tensor.shape[0]
            pH_fem = torch.full((npts, 1), 12.5)
            ch_fem = torch.full((npts, 1), 1.0)
            rpm_fem = torch.full((npts, 1), 1200.0)
            Q_fem = torch.full((npts, 1), 20e-3 / 3600)
            t_fem = torch.full((npts, 1), t_fix) / ts_fem

            X_fem_tensor = torch.cat([x_tensor, t_fem, pH_fem, ch_fem, rpm_fem, Q_fem], dim=1)
            return X_fem_tensor, c_tensor

        X_fem_NO, c_fem_NO_tensor = filter_and_create_fem_tensors(x_fem_NO, c_fem_NO, x_range)
        X_fem_OOH, c_fem_OOH_tensor = filter_and_create_fem_tensors(x_fem_OOH, c_fem_OOH, x_range)

        if X_fem_NO is not None:
            X_fem_NO = X_fem_NO.to(device)
        if c_fem_NO_tensor is not None:
            c_fem_NO_tensor = c_fem_NO_tensor.to(device)
        if X_fem_OOH is not None:
            X_fem_OOH = X_fem_OOH.to(device)
        if c_fem_OOH_tensor is not None:
            c_fem_OOH_tensor = c_fem_OOH_tensor.to(device)

    return (
        X_eq.to(device),
        X_ic.to(device),
        X_bc_left,
        X_bc_right,
        X_fem_NO,
        X_fem_OOH,
        c_fem_NO_tensor,
        c_fem_OOH_tensor,
    )


def multi_task_loss(model,X_eq,X_ic,X_bc_left,X_bc_right,X_fem_NO,X_fem_OOH,cno,r,r2,r1,u1,u2,Hno,Pno,T,h,nt_eta,exp_data,k=103.4,use_cfd=True,use_eta=True,device=None,):
    """
    Multi-condition PINN loss with normalized coordinates:
      x = x* (d/2), t = t* ts.
    PDE:
      cNO_t = D_NO cNO_xx - k cNO cOOH
      cOOH_t = D_OOH cOOH_xx - k cNO cOOH
    BC/IC and optional CFD + experimental removal loss are included.
    """
    X_eq.requires_grad_(True)
    X_ic.requires_grad_(True)
    if X_bc_left is not None:
        X_bc_left.requires_grad_(True)
    if X_bc_right is not None:
        X_bc_right.requires_grad_(True)
    if X_fem_NO is not None and use_cfd:
        X_fem_NO.requires_grad_(True)
    if X_fem_OOH is not None and use_cfd:
        X_fem_OOH.requires_grad_(True)

    # PDE residuals
    rpm_eq = X_eq[:, 4:5]
    Q_eq = X_eq[:, 5:6]
    cNO_eq, cOOH_eq = model(X_eq)

    d_val = get_d(rpm_eq, Q_eq, r=r)
    ts_val = get_ts(rpm_eq, Q_eq, r=r, r2=r2, r1=r1)

    De = get_de(rpm_eq)
    D_NO = 2.21e-9 * De
    D_OOH = 1.97e-9 * De

    cNO_t = partial_derivative_n(cNO_eq, X_eq, idx=1, order=1) / (ts_val + 1e-8)
    cOOH_t = partial_derivative_n(cOOH_eq, X_eq, idx=1, order=1) / (ts_val + 1e-8)
    cNO_xx = partial_derivative_n(cNO_eq, X_eq, idx=0, order=2) / ((d_val / 2) ** 2 + 1e-8)
    cOOH_xx = partial_derivative_n(cOOH_eq, X_eq, idx=0, order=2) / ((d_val / 2) ** 2 + 1e-8)

    res_NO = cNO_t - D_NO * cNO_xx + k * cNO_eq * cOOH_eq
    res_OOH = cOOH_t - D_OOH * cOOH_xx + k * cNO_eq * cOOH_eq
    L_pde_cNO = torch.mean(res_NO ** 2)
    L_pde_cOOH = torch.mean(res_OOH ** 2)
    L_pde = L_pde_cNO + L_pde_cOOH

    # IC: t* = 0 -> cNO=0, cOOH=0
    cNO_ic, cOOH_ic = model(X_ic)
    L_ic_cNO = torch.mean(cNO_ic ** 2)
    L_ic_cOOH = torch.mean(cOOH_ic ** 2)
    L_ic = L_ic_cNO + L_ic_cOOH

    # Left BC: x*=0 (NO Dirichlet, OOH Neumann)
    if X_bc_left is not None:
        pH_left = X_bc_left[:, 2:3]
        ch_left = X_bc_left[:, 3:4]
        rpm_left = X_bc_left[:, 4:5]
        Q_left = X_bc_left[:, 5:6]

        ooh = 1.0 / (1.0 + 10 ** (11.73 - pH_left)) * ch_left
        cNO_bc_left, cOOH_bc_left = model(X_bc_left)
        d_left = get_d(rpm_left, Q_left, r=r)

        L_bc_left_cNO = torch.mean((cNO_bc_left - cno) ** 2)
        cOOH_bc_left_x = partial_derivative_n(cOOH_bc_left, X_bc_left, idx=0, order=1)
        L_bc_left_cOOH = torch.mean((cOOH_bc_left_x / (d_left / 2)) ** 2)

        L_bc_left = L_bc_left_cNO + L_bc_left_cOOH
    else:
        L_bc_left_cNO = torch.tensor(0.0, device=X_eq.device)
        L_bc_left_cOOH = torch.tensor(0.0, device=X_eq.device)
        L_bc_left = torch.tensor(0.0, device=X_eq.device)

    # Right BC: x*=1 (NO Neumann, OOH Dirichlet)
    if X_bc_right is not None:
        pH_right = X_bc_right[:, 2:3]
        ch_right = X_bc_right[:, 3:4]
        rpm_right = X_bc_right[:, 4:5]
        Q_right = X_bc_right[:, 5:6]

        ooh = 1.0 / (1.0 + 10 ** (11.73 - pH_right)) * ch_right
        cNO_bc_right, cOOH_bc_right = model(X_bc_right)
        d_right = get_d(rpm_right, Q_right, r=r)

        cNO_bc_right_x = partial_derivative_n(cNO_bc_right, X_bc_right, idx=0, order=1)
        L_bc_right_cNO = torch.mean((cNO_bc_right_x / (d_right / 2)) ** 2)
        L_bc_right_cOOH = torch.mean((cOOH_bc_right - ooh) ** 2)

        L_bc_right = L_bc_right_cNO + L_bc_right_cOOH
    else:
        L_bc_right_cNO = torch.tensor(0.0, device=X_eq.device)
        L_bc_right_cOOH = torch.tensor(0.0, device=X_eq.device)
        L_bc_right = torch.tensor(0.0, device=X_eq.device)

    # Optional CFD data loss
    if X_fem_NO is not None and X_fem_OOH is not None and use_cfd:
        cNO_fem, _ = model(X_fem_NO)
        L_fem_cNO = torch.mean((cNO_fem - u1) ** 2)
        _, cOOH_fem = model(X_fem_OOH)
        L_fem_cOOH = torch.mean((cOOH_fem - u2) ** 2)
        L_fem = L_fem_cNO + L_fem_cOOH
    else:
        L_fem_cNO = torch.tensor(0.0, device=X_eq.device)
        L_fem_cOOH = torch.tensor(0.0, device=X_eq.device)
        L_fem = torch.tensor(0.0, device=X_eq.device)

    L_bc = L_bc_left + L_bc_right

    # Optional experimental removal-rate loss
    if use_eta:
        L_etaNO = eta_NO_batch_loss(
            model, r, r2, r1, cno, Hno, Pno, T, h,
            exp_data=exp_data, nt=nt_eta, device=device
        )
    else:
        L_etaNO = torch.tensor(0.0, device=device)

    L_cNO = L_pde_cNO + L_ic_cNO + L_bc_left_cNO + L_bc_right_cNO + L_fem_cNO + L_etaNO
    L_cOOH = L_pde_cOOH + L_ic_cOOH + L_bc_left_cOOH + L_bc_right_cOOH + L_fem_cOOH
    L_phys = L_cNO + L_cOOH - L_fem

    return (
        L_pde, L_ic, L_bc,
        L_pde_cNO, L_pde_cOOH,
        L_ic_cNO, L_ic_cOOH,
        L_bc_left_cNO, L_bc_left_cOOH,
        L_bc_right_cNO, L_bc_right_cOOH,
        L_cNO, L_cOOH,
        L_fem, L_etaNO, L_phys
    )

def load_exp_data(data_path="data.xlsx"):
    """Load experimental data from Excel; parse simple string expressions and normalize eta."""
    df = pd.read_excel(data_path)
    for col in ["Q", "G_N2", "N", "c", "pH", "yno_in"]:
        df[col] = df[col].apply(lambda x: eval(str(x)))
    df["eta"] = df["eta"] / 100.0
    return df.to_dict(orient="records")

def eta_NO_single_condition(model,pH, ch, rpm, Q, G_N2, y_NO_in,r, r2, r1, cno, Hno, Pno, T, h,nt=100,device=None,):
    """Predict NO removal (eta) for a single operating condition (differentiable w.r.t. model params)."""
    if device is None:
        device = next(model.parameters()).device

    # Scalars -> tensors (no gradients needed for these constants)
    pH_t = torch.tensor(pH, dtype=torch.float32, device=device).view(1, 1)
    ch_t = torch.tensor(ch, dtype=torch.float32, device=device).view(1, 1)
    rpm_t = torch.tensor(rpm, dtype=torch.float32, device=device).view(1, 1)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device).view(1, 1)
    G_N2_t = torch.tensor(G_N2, dtype=torch.float32, device=device)
    y_in_t = torch.tensor(y_NO_in, dtype=torch.float32, device=device)
    r_t = torch.tensor(r, dtype=torch.float32, device=device)
    r1_t = torch.tensor(r1, dtype=torch.float32, device=device)
    r2_t = torch.tensor(r2, dtype=torch.float32, device=device)
    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    cno_t = torch.tensor(cno, dtype=torch.float32, device=device)
    Hno_t = torch.tensor(Hno, dtype=torch.float32, device=device)
    Pno_t = torch.tensor(Pno, dtype=torch.float32, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)

    # Condition-dependent properties
    d = get_d(rpm_t, Q_t, r_t)
    ts = get_ts(rpm_t, Q_t, r_t, r2_t, r1_t)
    De = get_de(rpm_t)
    D1 = 2.21e-9 * De

    holdup = get_holdup(rpm_t, Q_t, r_t)
    a_bi = 6.0 * holdup / d

    # Time grid and boundary input at x=0
    t_phys = torch.linspace(0.0, ts.item(), nt, device=device).view(-1, 1)
    x_boundary = torch.zeros_like(t_phys, requires_grad=True)

    pH_vec = pH_t.expand_as(t_phys)
    ch_vec = ch_t.expand_as(t_phys)
    rpm_vec = rpm_t.expand_as(t_phys)
    Q_vec = Q_t.expand_as(t_phys)
    d_vec = d.expand_as(t_phys)
    ts_vec = ts.expand_as(t_phys)

    x_star = x_boundary / (d_vec / 2.0)
    t_star = t_phys / ts_vec

    X_input_boundary = torch.cat([x_star, t_star, pH_vec, ch_vec, rpm_vec, Q_vec], dim=1)
    X_input_boundary.requires_grad_(True)

    # Flux at x=0
    cNO_boundary, _ = model(X_input_boundary)
    dcNO_dx = torch.autograd.grad(
        outputs=cNO_boundary,
        inputs=x_boundary,
        grad_outputs=torch.ones_like(cNO_boundary),
        create_graph=True,
    )[0]
    flux_t = -D1 * dcNO_dx

    # Integrate flux over time
    dt = ts / (nt - 1)
    k = torch.trapz(flux_t.squeeze(-1), dx=dt.squeeze())

    # Mass-transfer correlations
    KL = k / (ts * cno_t)
    kg1 = torch.tensor(0.007954, device=device)
    Kg = 1.0 / (1.0 / kg1 + Hno_t / KL)
    ky_cal = 0.082 * T_t * (Pno_t * Kg)

    area = torch.pi * h_t * (r2_t ** 2 - r1_t ** 2)
    a1 = (1.0 - y_in_t) / y_in_t
    a2 = a_bi.squeeze() * area / G_N2_t

    exponent = a2 * ky_cal
    exp_mask = exponent > 700.0
    safe_exp = torch.exp(torch.clamp(exponent, max=700.0))
    y_out = torch.where(exp_mask, torch.zeros_like(exponent), 1.0 / (a1 * safe_exp + 1.0))

    eta_pred = 1.0 - y_out / y_in_t
    return eta_pred.squeeze()

def eta_NO_batch_loss(model, r, r2, r1, cno, Hno, Pno, T, h, exp_data, nt=100, device=None):
    """MSE loss over a batch of experimental conditions."""
    if device is None:
        device = next(model.parameters()).device

    eta_losses = []
    for row in exp_data:
        eta_pred_i = eta_NO_single_condition(
            model,
            pH=row["pH"], ch=row["c"],
            rpm=row["N"], Q=row["Q"],
            G_N2=row["G_N2"], y_NO_in=row["yno_in"],
            r=r, r2=r2, r1=r1,
            cno=cno, Hno=Hno, Pno=Pno, T=T, h=h,
            nt=nt, device=device,
        )
        eta_true_i = torch.tensor(row["eta"], dtype=torch.float32, device=device)
        eta_losses.append((eta_pred_i - eta_true_i) ** 2)

    if len(eta_losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.mean(torch.stack(eta_losses))

def gradnorm_update(model, task_losses: torch.Tensor, initial_losses: np.ndarray, alpha: float):
    """
    GradNorm update for task weights.
    task_losses: unweighted losses for each task.
    """
    shared_params = model.get_shared_layer()

    norms = []
    for i in range(len(task_losses)):
        gygw = torch.autograd.grad(
            task_losses[i],
            shared_params,
            retain_graph=True,
            allow_unused=True,
        )
        grad_flat = torch.cat([g.view(-1) for g in gygw if g is not None])
        norms.append(torch.norm(grad_flat, p=2))
    norms = torch.stack(norms)

    current_losses = task_losses.detach().cpu().numpy()
    ratio = current_losses / (initial_losses + 1e-8)
    ratio = torch.from_numpy(ratio).float().to(task_losses.device)

    mean_norm = torch.mean(norms)
    target = (mean_norm * (ratio ** alpha)).detach()

    weights = model.get_weights()
    grad_norm_loss = torch.sum(torch.abs(weights * norms - target))

    model.weights.grad = None
    grad = torch.autograd.grad(
        grad_norm_loss,
        model.weights,
        retain_graph=True,
        allow_unused=True,
    )
    if grad[0] is not None:
        model.weights.grad = grad[0]

    return grad_norm_loss


def get_dynamic_alpha(epoch: int, total_epochs: int) -> float:
    """Piecewise alpha schedule."""
    ratio = epoch / total_epochs
    if ratio < 0.3:
        return 1.5
    if ratio < 0.7:
        return 1.0
    return 0.5

class EMA:
    """Exponential moving average for scalar metrics."""

    def __init__(self, alpha=0.1, eps=1e-8):
        self.alpha = alpha
        self.ema = None
        self.eps = eps

    def update(self, new_value):
        if self.ema is None:
            self.ema = new_value + self.eps
        else:
            self.ema = self.alpha * new_value + (1 - self.alpha) * self.ema
        return self.ema

def predict(model,nx,nt,pH,ch,rpm,Q,r,r2,r1,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),apply_zero_correction=True,calculate_rate=False,):
    """
    Predict concentration fields (cNO, cOOH) on a space-time grid.
    Optionally compute NO flux rate at x=0 over time.
    """
    model.eval()

    d = get_d(rpm, Q, r)
    ts = get_ts(rpm, Q, r, r2, r1)

    x_phys = np.linspace(0, d / 2, nx).reshape(-1, 1)
    t_phys = np.linspace(0, ts, nt).reshape(-1, 1)

    X_phys, T_phys = np.meshgrid(x_phys, t_phys)
    X_flat = X_phys.flatten()[:, None]
    T_flat = T_phys.flatten()[:, None]

    pH_vals = np.full_like(X_flat, pH)
    ch_vals = np.full_like(X_flat, ch)
    rpm_vals = np.full_like(X_flat, rpm)
    Q_vals = np.full_like(X_flat, Q)

    def to_tensor(arr):
        return torch.from_numpy(arr).float().to(device).reshape(-1, 1)

    X_tensor = to_tensor(X_flat)
    T_tensor = to_tensor(T_flat)
    pH_tensor = to_tensor(pH_vals)
    ch_tensor = to_tensor(ch_vals)
    rpm_tensor = to_tensor(rpm_vals)
    Q_tensor = to_tensor(Q_vals)

    d_tensor = to_tensor(np.array([[d]]))
    ts_tensor = to_tensor(np.array([[ts]]))
    x_star = X_tensor / (d_tensor / 2)
    t_star = T_tensor / ts_tensor

    X_input = torch.cat([x_star, t_star, pH_tensor, ch_tensor, rpm_tensor, Q_tensor], dim=1)

    with torch.no_grad():
        cNO, cOOH = model(X_input)

    cNO_pred = cNO.cpu().numpy().reshape(nt, nx)
    cOOH_pred = cOOH.cpu().numpy().reshape(nt, nx)

    if apply_zero_correction:
        cNO_pred[0, 1:] = 0
        cOOH_pred[0, 1:] = 0

    if not calculate_rate:
        return x_phys, t_phys, cNO_pred, cOOH_pred

    # Flux rate at x=0
    De = get_de(rpm)
    D_NO = 2.21e-9 * De

    x_at_boundary = np.zeros_like(t_phys)
    X_tensor_boundary = to_tensor(x_at_boundary)
    X_tensor_boundary.requires_grad_(True)

    T_tensor_boundary = to_tensor(t_phys)
    pH_tensor_boundary = torch.full_like(T_tensor_boundary, pH)
    ch_tensor_boundary = torch.full_like(T_tensor_boundary, ch)
    rpm_tensor_boundary = torch.full_like(T_tensor_boundary, rpm)
    Q_tensor_boundary = torch.full_like(T_tensor_boundary, Q)

    x_star_boundary = X_tensor_boundary / (d_tensor / 2)
    t_star_boundary = T_tensor_boundary / ts_tensor

    X_input_boundary = torch.cat(
        [x_star_boundary, t_star_boundary, pH_tensor_boundary, ch_tensor_boundary, rpm_tensor_boundary, Q_tensor_boundary],
        dim=1,
    )

    cNO_boundary, _ = model(X_input_boundary)
    dcNO_dx = torch.autograd.grad(
        outputs=cNO_boundary,
        inputs=X_tensor_boundary,
        grad_outputs=torch.ones_like(cNO_boundary),
        create_graph=False,
    )[0]

    R_NO_tensor = -D_NO * dcNO_dx
    R_NO_numpy = R_NO_tensor.cpu().detach().numpy()

    return x_phys, t_phys, cNO_pred, cOOH_pred, R_NO_numpy

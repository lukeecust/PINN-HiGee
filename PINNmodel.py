import numpy as np
from numpy import pi, exp, log, sqrt, diff
from scipy.interpolate import interp1d
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm, Viewer
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import random
import warnings
from math import pi, log
import mpmath as mp
import pandas as pd


# 设置随机数种子
def setup_seed(seed):
    np.random.seed(seed) # 为numpy设置随机数种子
    torch.manual_seed(seed) # 为CPU设置随机数种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机数种子
    torch.backends.cudnn.deterministic = True # 确保使用确定性的算法，可能会降低性能，但保证结果一致
    random.seed(seed)
    torch.backends.cudnn.benchmark = False

# 定义计算梯度的函数
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# 定义计算偏导的函数
def partial_derivative_n(u, X, idx, order=1):
    """
    计算网络输出 u(X) 关于 X[:, idx] 的 n阶偏导数。

    :param u: 网络输出，可以是标量或形状为 [batch_size, 1] 的张量。
    :param X: 输入张量，形状为 [batch_size, dim]。
    :param idx: 对哪一个维度求导数，0 表示第一个维度，1 表示第二个维度，以此类推。
    :param order: 求导的阶数，默认为 1。
    :return: 对应于 X[:, idx] 的 n阶偏导张量，形状与 u 相同 ([batch_size, 1] 或类似形状)。

    示例：
    ```python
    # 假设 u 是网络的输出，X 是输入张量，要求 X 的第 0 维的一阶导数
    derivative = partial_derivative_n(u, X, idx=0, order=1)
    ```
    """
    grad = u
    for _ in range(order):
        grad = torch.autograd.grad(
            grad, X, grad_outputs=torch.ones_like(grad),
            create_graph=True, retain_graph=True
        )[0][:, idx:idx + 1]
    return grad

# 定义物理性质函数
def get_d(rpm, Q, r):
    """
    计算液滴直径 d，公式：
      d = 12.84 * ((72.0e-2)/(r * omega**2 * 1000.0))**0.630 * ((Q*3600)**0.201)
    输入：rpm, Q 均为 Tensor，支持广播
    返回：d, 形状与 rpm/Q 相同
    """
    omega = 2 * pi * rpm / 60.0
    d = (12.84 * ((72.0e-2) / (r * omega ** 2 * 1000.0)) ** 0.630 * ((Q * 3600) ** 0.201))
    return d

def get_ts(rpm, Q, r, r2, r1):
    """
    计算更新时间 ts，公式： ts = (r2 - r1)/(u0 * 31.0)
      其中 u0 = 0.02107 * Q**0.2279 * (omega**2 * r)
    """
    omega = 2 * pi * rpm / 60.0
    u0 = 0.02107 * Q ** 0.2279 * (omega ** 2 * r)
    ts = (r2 - r1) /(u0 * 31.0)
    return ts

def get_holdup(rpm, Q, r, v0 = 1.0e-6 , g0=100):
    """
    计算holdup值的函数
    
    返回:
    holdup - 无量纲参数
    """
    omega = 2 * pi * rpm / 60.0
    u0 = 0.02107 * Q ** 0.2279 * (omega ** 2 * r)
    holdup = 0.039 * ((r * omega**2 / g0)**(-0.5)) * ((u0 / 0.01)**0.6) * ((1.35e-6 / v0)**0.22)
    return holdup

rpm_data = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
De_data  = np.array([5.3064, 6.2257, 6.7727, 7.1618, 7.4629, 7.7081, 7.9144, 8.0818])
f_De = interp1d(rpm_data, De_data, kind='linear')  # 可选 'cubic' 等
def get_de(rpm):
    """给定任意 rpm，返回插值后的 De"""
    # 检查是否为张量
    if torch.is_tensor(rpm):
        # 转换为numpy进行检查和计算
        rpm_np = rpm.detach().cpu().numpy()

        # 检查范围
        if np.any(rpm_np < min(rpm_data)) or np.any(rpm_np > max(rpm_data)):
            raise ValueError("rpm 超出已有数据范围，需要外推或更多数据支持")

        # 使用numpy计算插值
        De_np = f_De(rpm_np)

        # 转回张量，保持设备和梯度属性
        De = torch.tensor(De_np, dtype=rpm.dtype, device=rpm.device)
        if rpm.requires_grad:
            De.requires_grad_()
        return De
    else:
        # 如果输入是标量，使用原来的逻辑
        if rpm < min(rpm_data) or rpm > max(rpm_data):
            raise ValueError("rpm 超出已有数据范围，需要外推或更多数据支持")
        return f_De(rpm)

# 定义共享层
class SharedLayer(nn.Module):
    def __init__(self, n_input=6, n_hidden=48, n_layers=4, activation=nn.Tanh):
        super(SharedLayer, self).__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
        layers.append(activation())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 定义 cNOPINN 和 cOOHPINN 的专属层
class cNOPINN(nn.Module):
    def __init__(self, n_hidden=32, n_output=1, n_layers=2, activation=nn.Tanh, dropout_prob=0.2):
        super(cNOPINN, self).__init__()
        layers = []
        layers.append(nn.Linear(n_hidden, n_hidden))
        nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
        layers.append(activation())
        layers.append(nn.Dropout(p=dropout_prob))  # 添加 Dropout
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_prob))  # 添加 Dropout
        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class cOOHPINN(nn.Module):
    def __init__(self, n_hidden=32, n_output=1, n_layers=2, activation=nn.Tanh, dropout_prob=0.2):
        super(cOOHPINN, self).__init__()
        layers = []
        layers.append(nn.Linear(n_hidden, n_hidden))
        nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
        layers.append(activation())
        layers.append(nn.Dropout(p=dropout_prob))  # 添加 Dropout
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight) # 对权重进行Xavier normal初始化
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout_prob))  # 添加 Dropout
        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)  # 对权重进行Xavier normal初始化
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 定义整个模型，包含共享层和两个专属子网络(任务数更改时，此处self.weights需要更改)
    
class CoupledPINN(nn.Module):
    def __init__(self,
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
                 dropout_prob_cOOH=0.2):
        super(CoupledPINN, self).__init__()

        # 保存标准化因子
        self.cNO_star = cNO_star
        self.cOOH_star = cOOH_star

        # 共享层
        self.shared = SharedLayer(
            n_input=6,
            n_hidden=n_hidden_shared,
            n_layers=n_layers_shared,
            activation=activation_shared,
        )

        # 子网络1: 预测 cNO(x,t)
        self.nn_cNO = cNOPINN(
            n_hidden=n_hidden_cNO,
            n_output=1,
            n_layers=n_layers_cNO,
            activation=activation_cNO,
            dropout_prob=dropout_prob_cNO
        )

        # 子网络2: 预测 cOOH(x,t)
        self.nn_cOOH = cOOHPINN(
            n_hidden=n_hidden_cOOH,
            n_output=1,
            n_layers=n_layers_cOOH,
            activation=activation_cOOH,
            dropout_prob=dropout_prob_cOOH
        )

        # 替换GradNorm权重为不确定度参数（初始化为0，相当于初始权重都是0.5）
        # self.log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=True)
        # GradNorm 可学习的任务权重
        self.weights = nn.Parameter(0.5413 * torch.ones(2, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        x_ = x[:, 0:1]
        t_ = x[:, 1:2]
        pH = x[:, 2:3]  # 提取 pH
        ch = x[:, 3:4]  # 提取 H2O2 浓度
        cOOH_star_dynamic = ch / (1.0 + 10 ** (11.73 - pH))
        shared_out = self.shared(x)
        cNO_hat = self.nn_cNO(shared_out)
        cOOH_hat = self.nn_cOOH(shared_out)
        cNO = (x_ - 1) ** 2 * (cNO_hat * x_ + (self.cNO_star)**0.5)**2
        cOOH = (x_) ** 2 * (cOOH_hat * (1 - x_) + (cOOH_star_dynamic)**0.5)**2
        # cNO = cNO_hat * x_ + self.cNO_star
        # cOOH = cOOH_hat * (1 - x_) + cOOH_star_dynamic
        # return cNO_hat, cOOH_hat
        return cNO, cOOH

    def get_shared_layer(self):
        # 仅返回共享层的参数
        return list(self.shared.parameters())

    # def get_weights(self):
    #     # 添加最小值限制，防止权重为0
    #     # 原始计算: 0.5 * torch.exp(-self.log_vars)
    #     # 修改后增加下限保护
    #     log_vars_clipped = torch.clamp(self.log_vars, max=10.0)  # 防止exp(-log_vars)变得过小
    #     weights = 0.5 * torch.exp(-log_vars_clipped)
    #     weights = torch.clamp(weights, min=1e-2)  # 确保权重不会小于1e-2
    #     return weights

    def get_weights(self):
        return torch.nn.functional.softplus(self.weights)

class SingleNetworkPINN(nn.Module):
    def __init__(self,
                 cNO_star,
                 cOOH_star,
                 n_input=6,
                 n_hidden=48,
                 n_layers=8,  # 总层数可以调整以匹配原模型的总深度
                 n_output=2,  # 关键：输出维度为2，分别对应两个任务
                 activation=nn.Tanh,
                 dropout_prob=0.2):
        super(SingleNetworkPINN, self).__init__()

        # 保存标准化因子
        self.cNO_star = cNO_star
        self.cOOH_star = cOOH_star

        # 构建一个单一的、端到端的网络
        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        nn.init.xavier_normal_(layers[-1].weight)
        layers.append(activation())
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))

        # 中间隐藏层
        # 注意：这里的 n_layers 代表了总的线性层数量
        for _ in range(n_layers - 2): # 减去输入层和输出层
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers.append(activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))

        # 输出层：直接输出两个值
        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 从输入中提取所需变量
        x_ = x[:, 0:1]
        t_ = x[:, 1:2]
        pH = x[:, 2:3]
        ch = x[:, 3:4]
        cOOH_star_dynamic = ch / (1.0 + 10 ** (11.73 - pH))

        # 单一网络直接输出两个预测值
        raw_outputs = self.net(x)

        # 将输出拆分给不同的任务
        cNO_hat = raw_outputs[:, 0:1]  # 第一个输出
        cOOH_hat = raw_outputs[:, 1:2] # 第二个输出

        # 应用与您原模型完全相同的物理约束（边界条件）
        # 这样做才能确保是在公平地比较网络架构本身
        cNO = (x_ - 1) ** 2 * (cNO_hat * x_ + (self.cNO_star)**0.5)**2
        cOOH = (x_) ** 2 * (cOOH_hat * (1 - x_) + (cOOH_star_dynamic)**0.5)**2
        
        return cNO, cOOH

# 首先，我们可以定义一个通用的“单任务网络”模块，方便复用
class SingleTaskNet(nn.Module):
    def __init__(self,
                 n_input=6,
                 n_hidden=128,
                 n_layers=13, # 这里的层数代表这个独立网络的总深度
                 n_output=1,  # 每个独立网络只输出一个值
                 activation=nn.Tanh,
                 dropout_prob=0.0):
        super(SingleTaskNet, self).__init__()

        layers = []
        layers.append(nn.Linear(n_input, n_hidden))
        nn.init.xavier_normal_(layers[-1].weight)
        layers.append(activation())
        if dropout_prob > 0:
            layers.append(nn.Dropout(p=dropout_prob))

        # 中间隐藏层
        for _ in range(n_layers - 2): # 减去输入层和输出层
            layers.append(nn.Linear(n_hidden, n_hidden))
            nn.init.xavier_normal_(layers[-1].weight)
            layers.append(activation())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))

        # 输出层
        layers.append(nn.Linear(n_hidden, n_output))
        nn.init.xavier_normal_(layers[-1].weight)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# 定义整个“独立网络”模型，它包含两个完全独立的子网络
class IndependentPINN(nn.Module):
    def __init__(self,
                 cNO_star,
                 cOOH_star,
                 # cNO 网络的参数
                 n_hidden_cNO=128,
                 n_layers_cNO=13,
                 activation_cNO=nn.Tanh,
                 dropout_prob_cNO=0.0,
                 # cOOH 网络的参数
                 n_hidden_cOOH=128,
                 n_layers_cOOH=13,
                 activation_cOOH=nn.Tanh,
                 dropout_prob_cOOH=0.0):
        super(IndependentPINN, self).__init__()

        # 保存标准化因子
        self.cNO_star = cNO_star
        self.cOOH_star = cOOH_star

        # 网络1: 独立预测 cNO
        self.net_cNO = SingleTaskNet(
            n_input=6,
            n_hidden=n_hidden_cNO,
            n_layers=n_layers_cNO,
            n_output=1,
            activation=activation_cNO,
            dropout_prob=dropout_prob_cNO
        )

        # 网络2: 独立预测 cOOH
        self.net_cOOH = SingleTaskNet(
            n_input=6,
            n_hidden=n_hidden_cOOH,
            n_layers=n_layers_cOOH,
            n_output=1,
            activation=activation_cOOH,
            dropout_prob=dropout_prob_cOOH
        )

    def forward(self, x):
        # 从输入中提取所需变量
        x_ = x[:, 0:1]
        t_ = x[:, 1:2]
        pH = x[:, 2:3]
        ch = x[:, 3:4]
        cOOH_star_dynamic = ch / (1.0 + 10 ** (11.73 - pH))

        # 分别通过两个独立的网络进行预测
        cNO_hat = self.net_cNO(x)
        cOOH_hat = self.net_cOOH(x)

        # 应用与您原模型完全相同的物理约束（边界条件）
        cNO = (x_ - 1) ** 2 * (cNO_hat * x_ + (self.cNO_star)**0.5)**2
        cOOH = (x_) ** 2 * (cOOH_hat * (1 - x_) + (cOOH_star_dynamic)**0.5)**2
        
        return cNO, cOOH

def print_grad_norm(model):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None and 'weights' not in name:  # 排除权重参数
            param_norm = param.grad.norm(2).item()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_clip_params(model):
    return [p for n, p in model.named_parameters() if p.grad is not None and 'weights' not in n]

# 生成采样点的函数
u1, u2 = [torch.tensor(np.loadtxt(f'u{i}x200.csv', delimiter=','), dtype=torch.float32) for i in (1,2)]

def generate_sampling_points(N=20000, N1=5000,
                             pH_range=(10.0, 13.0),
                             ch_range=(0.05, 1.6),
                             rpm_range=(200.0, 1600.0),
                             gn2=2.0 / 3600,
                             x_range=(0.0, 1.0),
                             t_range=(0.0, 1.0),
                             x_fem_NO=None, x_fem_OOH=None,
                             c_fem_NO=None, c_fem_OOH=None,
                             t_fix=1.0,
                             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                             sampling_method='sobol'):
    """
    生成多工况 PINN 的采样点，输入为6维：[x^*, t^*, pH, c_h, rpm, Q]
    
    参数:
    x_range, t_range: 可以是单个元组 (min, max) 或分段区间的列表 [(min1, max1), (min2, max2), ...]
    sampling_method: 采样方法，可选'random'或'sobol'（默认）
    """
    
    # 辅助函数：根据选择的方法生成[0,1)区间的采样点
    def generate_samples(dim, num_samples, method):
        if num_samples <= 0:
            return torch.empty(0, dim)
        if method == 'random':
            return torch.rand(num_samples, dim)
        elif method == 'sobol':
            try:
                soboleng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
                return soboleng.draw(num_samples)
            except Exception as e:
                print(f"警告: 无法使用PyTorch的SobolEngine: {e}")
                try:
                    import sobol_seq
                    return torch.tensor(sobol_seq.i4_sobol_generate(dim, num_samples), dtype=torch.float32)
                except ImportError:
                    print("警告: 无法使用Sobol序列，回退到随机采样")
                    return torch.rand(num_samples, dim)
        else:
            print(f"警告: 不支持的采样方法: {method}，使用随机采样")
            return torch.rand(num_samples, dim)

    # --- 新增：处理分段区间的辅助函数 ---
    def _sample_from_intervals(num_samples, intervals, method):
        """从一个或多个区间进行采样"""
        if num_samples <= 0:
            return torch.empty(0, 1)
            
        lengths = [end - start for start, end in intervals]
        total_length = sum(lengths)
        
        if total_length <= 1e-9: # 避免除以零
            # 如果总长度为0，无法采样，返回一个空张量或在某个点采样
            # 这里我们选择在第一个区间的起点采样
            return torch.full((num_samples, 1), intervals[0][0])

        samples_per_interval = [int(np.round(num_samples * length / total_length)) for length in lengths]
        diff = num_samples - sum(samples_per_interval)
        if diff != 0:
            idx_max_len = np.argmax(lengths)
            samples_per_interval[idx_max_len] += diff

        all_samples = []
        for i, (start, end) in enumerate(intervals):
            n_sub_samples = samples_per_interval[i]
            if n_sub_samples > 0:
                p = generate_samples(1, n_sub_samples, method)
                sub_samples = start + (end - start) * p
                all_samples.append(sub_samples)
        
        if not all_samples:
            return torch.empty(0, 1)
            
        result = torch.cat(all_samples, dim=0)
        # 打乱顺序以确保随机性
        return result[torch.randperm(result.shape[0])]

    # --- 标准化输入，兼容单个元组和列表 ---
    if isinstance(x_range, tuple): x_range = [x_range]
    if isinstance(t_range, tuple): t_range = [t_range]

    Q_range = (gn2 / 120, gn2 / 30)

    # PDE 内部点
    p = generate_samples(4, N, sampling_method)
    pH_vals = pH_range[0] + (pH_range[1] - pH_range[0]) * p[:, 0:1]
    ch_vals = ch_range[0] + (ch_range[1] - ch_range[0]) * p[:, 1:2]
    rpm_vals = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p[:, 2:3]
    Q_vals = Q_range[0] + (Q_range[1] - Q_range[0]) * p[:, 3:4]
    
    x_vals = _sample_from_intervals(N, x_range, sampling_method)
    t_vals = _sample_from_intervals(N, t_range, sampling_method)

    X_eq = torch.cat([x_vals, t_vals, pH_vals, ch_vals, rpm_vals, Q_vals], dim=1)

    # 初始条件点: t=0
    num_ic_samples = N1 // 3
    p_ic = generate_samples(4, num_ic_samples, sampling_method)
    pH_ic = pH_range[0] + (pH_range[1] - pH_range[0]) * p_ic[:, 0:1]
    ch_ic = ch_range[0] + (ch_range[1] - ch_range[0]) * p_ic[:, 1:2]
    rpm_ic = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p_ic[:, 2:3]
    Q_ic = Q_range[0] + (Q_range[1] - Q_range[0]) * p_ic[:, 3:4]

    x_ic = _sample_from_intervals(num_ic_samples, x_range, sampling_method)
    t_ic = torch.full_like(x_ic, min(start for start, end in t_range)) # t=t_min

    X_ic = torch.cat([x_ic, t_ic, pH_ic, ch_ic, rpm_ic, Q_ic], dim=1)

    # 边界条件点
    X_bc_left = None
    X_bc_right = None
    num_bc_samples = N1 // 3
    
    global_min_x = min(start for start, end in x_range)
    global_max_x = max(end for start, end in x_range)
    
    # 左边界 x=0
    if global_min_x <= 1e-9: # 使用小容差避免浮点数问题
        p_left = generate_samples(4, num_bc_samples, sampling_method)
        pH_left = pH_range[0] + (pH_range[1] - pH_range[0]) * p_left[:, 0:1]
        ch_left = ch_range[0] + (ch_range[1] - ch_range[0]) * p_left[:, 1:2]
        rpm_left = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p_left[:, 2:3]
        Q_left = Q_range[0] + (Q_range[1] - Q_range[0]) * p_left[:, 3:4]

        t_left = _sample_from_intervals(num_bc_samples, t_range, sampling_method)
        x_left = torch.zeros_like(t_left)
        X_bc_left = torch.cat([x_left, t_left, pH_left, ch_left, rpm_left, Q_left], dim=1).to(device)

    # 右边界 x=1
    if global_max_x >= 1.0 - 1e-9:
        p_right = generate_samples(4, num_bc_samples, sampling_method)
        pH_right = pH_range[0] + (pH_range[1] - pH_range[0]) * p_right[:, 0:1]
        ch_right = ch_range[0] + (ch_range[1] - ch_range[0]) * p_right[:, 1:2]
        rpm_right = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * p_right[:, 2:3]
        Q_right = Q_range[0] + (Q_range[1] - Q_range[0]) * p_right[:, 3:4]
        
        t_right = _sample_from_intervals(num_bc_samples, t_range, sampling_method)
        x_right = torch.ones_like(t_right)
        X_bc_right = torch.cat([x_right, t_right, pH_right, ch_right, rpm_right, Q_right], dim=1).to(device)

    # CFD数据点处理
    X_fem_NO, X_fem_OOH = None, None
    c_fem_NO_tensor, c_fem_OOH_tensor = None, None
    
    if x_fem_NO is not None and c_fem_NO is not None and x_fem_OOH is not None and c_fem_OOH is not None:
        d_fem = get_d(rpm=1200, Q=20e-3/3600, r=0.052802)
        ts_fem = get_ts(rpm=1200, Q=20e-3/3600, r=0.052802, r2=0.07, r1=0.026)

        def filter_and_create_fem_tensors(x_fem_raw, c_fem_raw, intervals):
            if torch.is_tensor(x_fem_raw): x_fem_array = x_fem_raw.cpu().numpy()
            else: x_fem_array = np.array(x_fem_raw)
            x_fem_norm = x_fem_array / (d_fem/2)
            
            combined_mask = np.zeros_like(x_fem_norm, dtype=bool)
            for start, end in intervals:
                interval_mask = (x_fem_norm >= start) & (x_fem_norm <= end)
                combined_mask = np.logical_or(combined_mask, interval_mask)
            
            if not np.any(combined_mask): return None, None

            x_fem_filtered = x_fem_norm[combined_mask]
            
            if torch.is_tensor(c_fem_raw): c_fem_filtered = c_fem_raw.cpu().numpy()[combined_mask]
            else: c_fem_filtered = np.array(c_fem_raw)[combined_mask]
                
            x_tensor = torch.tensor(x_fem_filtered, dtype=torch.float32).reshape(-1, 1)
            c_tensor = torch.tensor(c_fem_filtered, dtype=torch.float32).reshape(-1, 1)
            
            num_pts = x_tensor.shape[0]
            pH_fem = torch.full((num_pts, 1), 12.5)
            ch_fem = torch.full((num_pts, 1), 1.0)
            rpm_fem = torch.full((num_pts, 1), 1200.0)
            Q_fem = torch.full((num_pts, 1), 20e-3/3600)
            t_fem = torch.full((num_pts, 1), t_fix) / ts_fem
            
            X_fem_tensor = torch.cat([x_tensor, t_fem, pH_fem, ch_fem, rpm_fem, Q_fem], dim=1)
            return X_fem_tensor, c_tensor

        X_fem_NO, c_fem_NO_tensor = filter_and_create_fem_tensors(x_fem_NO, c_fem_NO, x_range)
        X_fem_OOH, c_fem_OOH_tensor = filter_and_create_fem_tensors(x_fem_OOH, c_fem_OOH, x_range)
        
        if X_fem_NO is not None: X_fem_NO = X_fem_NO.to(device)
        if c_fem_NO_tensor is not None: c_fem_NO_tensor = c_fem_NO_tensor.to(device)
        if X_fem_OOH is not None: X_fem_OOH = X_fem_OOH.to(device)
        if c_fem_OOH_tensor is not None: c_fem_OOH_tensor = c_fem_OOH_tensor.to(device)

    # 移动到设备
    X_eq = X_eq.to(device)
    X_ic = X_ic.to(device)

    return X_eq, X_ic, X_bc_left, X_bc_right, X_fem_NO, X_fem_OOH, c_fem_NO_tensor, c_fem_OOH_tensor


def multi_task_loss(model, X_eq, X_ic, X_bc_left, X_bc_right, X_fem_NO, X_fem_OOH, cno, r, r2, r1, u1, u2, Hno, Pno, T, h, nt_eta, exp_data, k=103.4, use_cfd=True, use_eta=True, device=None):
    """
    对于多工况 PINN，考虑物理域归一化：
      物理 x = x^* * (d/2), 物理 t = t^* * ts
    其中 d 和 ts 是根据工况参数 (rpm, Q) 通过公式计算得到。
    PDE:
      cNO_t = D_NO * cNO_xx - k * cNO * cOOH
      cOOH_t = D_OOH * cOOH_xx - k * cNO * cOOH
    初始条件：t^* = 0  => cNO = 0, cOOH = 0
    
    边界条件：
      NO:  左边界 x^* = 0  => cNO = cNO_bc (Dirichlet)
           右边界 x^* = 1  => ∂cNO/∂x = 0 (Neumann)
      OOH: 左边界 x^* = 0  => ∂cOOH/∂x = 0 (Neumann)  
           右边界 x^* = 1  => cOOH = cOOH_bc (Dirichlet)
    
    k 为反应速率常数（可设为常数或依赖工况）。
    """
    # 开启自动微分
    X_eq.requires_grad_(True)
    X_ic.requires_grad_(True)
    # 仅当边界点不为None时才设置requires_grad
    if X_bc_left is not None:
        X_bc_left.requires_grad_(True)
    if X_bc_right is not None:
        X_bc_right.requires_grad_(True)
    if X_fem_NO is not None and use_cfd:
        X_fem_NO.requires_grad_(True)
    if X_fem_OOH is not None and use_cfd:
        X_fem_OOH.requires_grad_(True)
    
    # ---- 1) PDE 残差计算 ----
    # 分离 PDE 点的输入
    rpm_eq = X_eq[:, 4:5]
    Q_eq = X_eq[:, 5:6]
    # 前向网络
    cNO_eq, cOOH_eq = model(X_eq)
    # 根据工况动态计算物理域参数：
    d_val = get_d(rpm_eq, Q_eq, r=r)  # 液滴直径
    ts_val = get_ts(rpm_eq, Q_eq, r=r, r2=r2, r1=r1)  # 更新时间
    # 计算扩散系数
    De = get_de(rpm_eq)
    D_NO = 2.21e-9 * De
    D_OOH = 1.97e-9 * De
    # 计算归一化下物理导数：
    # 物理 t = t^* * ts  => ∂/∂t_physical = (1/ts) ∂/∂t^*
    cNO_t = partial_derivative_n(cNO_eq, X_eq, idx=1, order=1)/(ts_val + 1e-8)
    cOOH_t = partial_derivative_n(cOOH_eq, X_eq, idx=1, order=1)/(ts_val + 1e-8)
    # 物理 x = x^* * (d/2)  => ∂^2/∂x_phys^2 = (1/ (d/2)^2) ∂^2/∂(x^*)^2
    cNO_xx = partial_derivative_n(cNO_eq, X_eq, idx=0, order=2)/((d_val/2)**2 + 1e-8)
    cOOH_xx = partial_derivative_n(cOOH_eq, X_eq, idx=0, order=2)/((d_val/2)**2 + 1e-8)
    # PDE 残差：注意反应项符号，这里用 + k*cNO*cOOH（如物理公式所示）
    res_NO = cNO_t - D_NO * cNO_xx + k * cNO_eq * cOOH_eq
    res_OOH = cOOH_t - D_OOH * cOOH_xx + k * cNO_eq * cOOH_eq
    L_pde_cNO = torch.mean(res_NO ** 2)
    L_pde_cOOH = torch.mean(res_OOH ** 2)
    L_pde = L_pde_cNO + L_pde_cOOH
    
    # ---- 2) 初始条件: t^* = 0  ----
    # 期望 cNO = 0, cOOH = 0
    cNO_ic, cOOH_ic = model(X_ic)
    L_ic_cNO = torch.mean(cNO_ic ** 2)
    L_ic_cOOH = torch.mean(cOOH_ic ** 2)
    L_ic = torch.mean(cNO_ic ** 2) + torch.mean(cOOH_ic ** 2)
    
    # ---- 左边界: x^* = 0  ----
    if X_bc_left is not None:
        # 提取左边界点输入
        pH_left = X_bc_left[:, 2:3]
        ch_left = X_bc_left[:, 3:4]
        rpm_left = X_bc_left[:, 4:5]
        Q_left = X_bc_left[:, 5:6]
        
        # 根据公式计算边界值：
        ooh = 1.0 / (1.0 + 10 ** (11.73 - pH_left)) * ch_left
        cNO_bc_left, cOOH_bc_left = model(X_bc_left)
        d_left = get_d(rpm_left, Q_left, r=r)
        
        # NO: 左边界 Dirichlet 条件 cNO = cno
        L_bc_left_cNO = torch.mean((cNO_bc_left - cno) ** 2)
        
        # OOH: 左边界 Neumann 条件 ∂cOOH/∂x = 0
        cOOH_bc_left_x = partial_derivative_n(cOOH_bc_left, X_bc_left, idx=0, order=1)
        L_bc_left_cOOH = torch.mean((cOOH_bc_left_x / (d_left / 2)) ** 2)
        
        L_bc_left = L_bc_left_cNO + L_bc_left_cOOH
    else:
        # 如果左边界点为None，将对应损失设为0
        L_bc_left_cNO = torch.tensor(0.0, device=X_eq.device)
        L_bc_left_cOOH = torch.tensor(0.0, device=X_eq.device)
        L_bc_left = torch.tensor(0.0, device=X_eq.device)
    
    # ---- 右边界: x^* = 1  ----
    if X_bc_right is not None:
        # 提取右边界点输入
        pH_right = X_bc_right[:, 2:3]
        ch_right = X_bc_right[:, 3:4]
        rpm_right = X_bc_right[:, 4:5]
        Q_right = X_bc_right[:, 5:6]
        
        # 根据公式计算边界值：
        ooh = 1.0 / (1.0 + 10 ** (11.73 - pH_right)) * ch_right
        cNO_bc_right, cOOH_bc_right = model(X_bc_right)
        d_right = get_d(rpm_right, Q_right, r=r)
        
        # NO: 右边界 Neumann 条件 ∂cNO/∂x = 0
        cNO_bc_right_x = partial_derivative_n(cNO_bc_right, X_bc_right, idx=0, order=1)
        L_bc_right_cNO = torch.mean((cNO_bc_right_x / (d_right / 2)) ** 2)
        
        # OOH: 右边界 Dirichlet 条件 cOOH = ooh
        L_bc_right_cOOH = torch.mean((cOOH_bc_right - ooh) ** 2)
        
        L_bc_right = L_bc_right_cNO + L_bc_right_cOOH
    else:
        # 如果右边界点为None，将对应损失设为0
        L_bc_right_cNO = torch.tensor(0.0, device=X_eq.device)
        L_bc_right_cOOH = torch.tensor(0.0, device=X_eq.device)
        L_bc_right = torch.tensor(0.0, device=X_eq.device)
    
    # ---- 3) fem条件  ----
    if X_fem_NO is not None and X_fem_OOH is not None and use_cfd:
        # 只有当use_cfd为True时才计算CFD数据损失
        cNO_fem, _ = model(X_fem_NO)
        L_fem_cNO = torch.mean((cNO_fem - u1) ** 2)
        _, cOOH_fem = model(X_fem_OOH)
        L_fem_cOOH = torch.mean((cOOH_fem - u2) ** 2)
        L_fem = L_fem_cNO + L_fem_cOOH
    else:
        # 不使用CFD数据或数据为None时，将CFD损失设为0
        L_fem_cNO = torch.tensor(0.0, device=X_eq.device)
        L_fem_cOOH = torch.tensor(0.0, device=X_eq.device)
        L_fem = torch.tensor(0.0, device=X_eq.device)
    
    L_bc = L_bc_left + L_bc_right

    # ---- 4) 实验脱除率损失 ----
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
    

    return L_pde, L_ic, L_bc, L_pde_cNO, L_pde_cOOH, L_ic_cNO, L_ic_cOOH, L_bc_left_cNO, L_bc_left_cOOH, L_bc_right_cNO, L_bc_right_cOOH, L_cNO, L_cOOH, L_fem, L_etaNO, L_phys

def load_exp_data(data_path="data.xlsx"):
    df = pd.read_excel(data_path)

    # 把类似 "20e-3/3600"、"1000*0.000001" 这样的字符串公式转成数值
    for col in ['Q', 'G_N2', 'N', 'c', 'pH', 'yno_in']:
        df[col] = df[col].apply(lambda x: eval(str(x)))

    # eta 从百分比转成 0~1
    df['eta'] = df['eta'] / 100.0

    # 转成 list[dict]，方便后面遍历
    exp_data = df.to_dict(orient='records')
    return exp_data


def eta_NO_single_condition(
    model,
    pH, ch, rpm, Q, G_N2, y_NO_in,
    r, r2, r1, cno, Hno, Pno, T, h,
    nt=100,
    device=None,
):
    """
    返回：eta_pred (标量 Tensor)，可用于构建 (eta_pred - eta_true)^2 的损失。
    """
    if device is None:
        device = next(model.parameters()).device

    # -------- 1. 标量物性参数转为 Tensor（不需要梯度） --------
    pH_t     = torch.tensor(pH,     dtype=torch.float32, device=device).view(1, 1)
    ch_t     = torch.tensor(ch,     dtype=torch.float32, device=device).view(1, 1)
    rpm_t    = torch.tensor(rpm,    dtype=torch.float32, device=device).view(1, 1)
    Q_t      = torch.tensor(Q,      dtype=torch.float32, device=device).view(1, 1)
    G_N2_t   = torch.tensor(G_N2,   dtype=torch.float32, device=device)
    y_in_t   = torch.tensor(y_NO_in,dtype=torch.float32, device=device)
    r_t      = torch.tensor(r,      dtype=torch.float32, device=device)
    r1_t     = torch.tensor(r1,     dtype=torch.float32, device=device)
    r2_t     = torch.tensor(r2,     dtype=torch.float32, device=device)
    h_t      = torch.tensor(h,      dtype=torch.float32, device=device)
    cno_t    = torch.tensor(cno,    dtype=torch.float32, device=device)
    Hno_t    = torch.tensor(Hno,    dtype=torch.float32, device=device)
    Pno_t    = torch.tensor(Pno,    dtype=torch.float32, device=device)
    T_t      = torch.tensor(T,      dtype=torch.float32, device=device)

    # -------- 2. 计算 d, ts, De, holdup 等（使用 torch 友好物性函数） --------
    d = get_d(rpm_t, Q_t, r_t)                     # (1,1)
    ts = get_ts(rpm_t, Q_t, r_t, r2_t, r1_t)       # (1,1)

    # 可用你现有的 get_de（numpy + interp1d）；对模型参数来说 De 是常数即可
    De = get_de(rpm_t)                             # (1,1)
    D1 = 2.21e-9 * De                              # (1,1)

    holdup = get_holdup(rpm_t, Q_t, r_t)           # (1,1)
    a_bi = 6.0 * holdup / d                        # (1,1)

    # -------- 3. 构造时间网格和边界输入 x=0 --------
    # t_phys: (nt, 1)，0 ~ ts
    t_phys = torch.linspace(0.0, ts.item(), nt, device=device).view(-1, 1)
    x_boundary = torch.zeros_like(t_phys, requires_grad=True)

    # 其他工况参数在时间维度上扩展
    pH_vec  = pH_t.expand_as(t_phys)
    ch_vec  = ch_t.expand_as(t_phys)
    rpm_vec = rpm_t.expand_as(t_phys)
    Q_vec   = Q_t.expand_as(t_phys)
    d_vec   = d.expand_as(t_phys)
    ts_vec  = ts.expand_as(t_phys)

    # 归一化坐标：x* = x / (d/2), t* = t / ts
    x_star = x_boundary / (d_vec / 2.0)
    t_star = t_phys / ts_vec

    # 组装模型输入: [x*, t*, pH, ch, rpm, Q]
    X_input_boundary = torch.cat(
        [x_star, t_star, pH_vec, ch_vec, rpm_vec, Q_vec], dim=1
    )
    X_input_boundary.requires_grad_(True)   # 为了对 x 求导

    # -------- 4. 前向 + 自动微分得到界面通量 J(t) --------
    cNO_boundary, _ = model(X_input_boundary)     # (nt, 1)

    # ∂cNO / ∂x_phys，在 x=0 处
    dcNO_dx = torch.autograd.grad(
        outputs=cNO_boundary,
        inputs=x_boundary,
        grad_outputs=torch.ones_like(cNO_boundary),
        create_graph=True       # 关键：保持对模型参数可微
    )[0]                        # (nt, 1)

    # J(t) = -D1 * ∂c/∂x
    flux_t = - D1 * dcNO_dx     # 广播到 (nt,1)

    # -------- 5. 时间积分得到 k = ∫ J(t) dt --------
    dt = ts / (nt - 1)
    k = torch.trapz(flux_t.squeeze(-1), dx=dt.squeeze())   # 标量

    # -------- 6. 根据原 merged_transport_solver 的公式继续算下去 --------
    KL = k / (ts * cno_t)                       # 液相传质系数
    kg1 = torch.tensor(0.007954, device=device)
    Kg = 1.0 / (1.0 / kg1 + Hno_t / KL)

    ky_cal = 0.082 * T_t * (Pno_t * Kg)

    # 这里的 a1, a2 与原代码保持一致
    area = torch.pi * h_t * (r2_t**2 - r1_t**2)     # π * h * (r2^2 - r1^2)
    a1 = (1.0 - y_in_t) / y_in_t
    a2 = a_bi.squeeze() * area / G_N2_t             # a_bi 是 (1,1)，挤掉多余维度

    exponent = a2 * ky_cal
    # 防溢出处理（与原逻辑近似）
    exp_mask = exponent > 700.0
    safe_exp = torch.exp(torch.clamp(exponent, max=700.0))
    y_out = torch.where(
        exp_mask,
        torch.zeros_like(exponent),
        1.0 / (a1 * safe_exp + 1.0)
    )

    eta_pred = 1.0 - y_out / y_in_t
    return eta_pred.squeeze()

def eta_NO_batch_loss(model, r, r2, r1, cno, Hno, Pno, T, h,
                      exp_data, nt=100, device=None):
    if device is None:
        device = next(model.parameters()).device

    eta_losses = []

    for row in exp_data:
        eta_pred_i = eta_NO_single_condition(
            model,
            pH=row['pH'], ch=row['c'],
            rpm=row['N'], Q=row['Q'],
            G_N2=row['G_N2'], y_NO_in=row['yno_in'],
            r=r, r2=r2, r1=r1,
            cno=cno, Hno=Hno, Pno=Pno, T=T, h=h,
            nt=nt, device=device
        )
        eta_true_i = torch.tensor(row['eta'], dtype=torch.float32, device=device)
        eta_losses.append((eta_pred_i - eta_true_i)**2)

    if len(eta_losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.mean(torch.stack(eta_losses))


# 定义 GradNorm 损失函数
def gradnorm_update(model,
                    task_losses: torch.Tensor,
                    initial_losses: np.ndarray,
                    alpha: float):
    """
    参数修正说明：
    - task_losses: shape=[8], 各任务的原始损失（未加权）
    - model.weights: 需通过Softplus保证非负性
    """
    # 1. 获取共享参数（确保包含所有可训练共享层）
    shared_params = model.get_shared_layer()

    # 2. 计算各任务的未加权梯度范数 G_i(t)
    norms = []
    for i in range(len(task_losses)):
        # 计算 ∇_W L_i(t)
        gygw = torch.autograd.grad(
            task_losses[i],
            shared_params,
            retain_graph=True,
            allow_unused=True  # 处理可能未使用的参数
        )
        # 拼接所有梯度并计算L2范数
        grad_flatten = torch.cat([g.view(-1) for g in gygw if g is not None])
        grad_norm_i = torch.norm(grad_flatten, p=2)
        norms.append(grad_norm_i)
    norms = torch.stack(norms)  # shape=[8]

    # 3. 计算 r_i(t) = L_i(t)/L_i(0) （禁用额外归一化）
    current_losses = task_losses.detach().cpu().numpy()
    ratio = current_losses / (initial_losses + 1e-8)  # 防止除零
    ratio = torch.from_numpy(ratio).float().to(task_losses.device)  # 保持原始比例

    # 4. 计算平均梯度范数与目标值
    mean_norm = torch.mean(norms)
    target = mean_norm * (ratio ** alpha)
    target = target.detach()  # 阻断梯度反向传播

    # 5. 计算梯度归一化损失
    weights = model.get_weights()  # 确保通过Softplus等激活函数约束非负
    grad_norm_loss = torch.sum(torch.abs(weights * norms - target))

    # 6. 计算权重梯度
    model.weights.grad = None  # 清空旧梯度
    grad = torch.autograd.grad(
        grad_norm_loss,
        model.weights,
        retain_graph=True,
        allow_unused=True
    )
    if grad[0] is not None:
        model.weights.grad = grad[0]

    return grad_norm_loss

# 定义动态 alpha 调整函数
def get_dynamic_alpha(epoch, total_epochs):
    """
    根据当前 epoch 所在区间，返回 alpha:
     - 前 30% : alpha=1.5
     - 中间 40%: alpha=1.0
     - 后 30% : alpha=0.5
    """
    ratio = epoch / total_epochs
    if ratio < 0.3:
        return 1.5
    elif ratio < 0.7:
        return 1.0
    else:
        return 0.5

# 定义EMA类
class EMA:
    def __init__(self, alpha=0.1, eps=1e-8):
        """
        初始化 EMA 对象
        :param alpha: EMA 的平滑因子
        :param eps: 防止初始为零的微小值
        """
        self.alpha = alpha
        self.ema = None
        self.eps = eps

    def update(self, new_value):
        """
        更新 EMA 值
        :param new_value: 新的数值
        :return: 更新后的 EMA 值
        """
        if self.ema is None:
            self.ema = new_value + self.eps
        else:
            self.ema = self.alpha * new_value + (1 - self.alpha) * self.ema
        return self.ema

def predict(model, nx, nt, pH, ch, rpm, Q, r, r2, r1,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            apply_zero_correction=True,
            calculate_rate=False): # <-- 新增参数
    """
    功能全面的预测函数：
    1. 预测整个时空域的浓度场 (cNO, cOOH)。
    2. (可选) 计算在 x=0 边界处，NO随时间变化的传质速率。

    参数:
    ... (原始参数不变) ...
    calculate_rate (bool): 如果为True，则额外计算并返回NO的传质速率。

    返回:
    如果 calculate_rate is False (默认):
        tuple: (x_phys, t_phys, cNO_pred, cOOH_pred)
    如果 calculate_rate is True:
        tuple: (x_phys, t_phys, cNO_pred, cOOH_pred, R_NO)
               R_NO 是一个 shape (nt, 1) 的传质速率数组。
    """
    model.eval() # 确保模型处于评估模式

    # =======================================================================
    # 步骤 1: 计算整个时空域的浓度场 (与原始函数逻辑相同)
    # =======================================================================
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

    # 为了效率，浓度场预测在 no_grad 环境下进行
    with torch.no_grad():
        cNO, cOOH = model(X_input)

    cNO_pred = cNO.cpu().numpy().reshape(nt, nx)
    cOOH_pred = cOOH.cpu().numpy().reshape(nt, nx)

    if apply_zero_correction:
        cNO_pred[0, 1:] = 0
        cOOH_pred[0, 1:] = 0

    # =======================================================================
    # 步骤 2: (可选) 如果需要，计算传质速率
    # =======================================================================
    if not calculate_rate:
        # 如果不计算速率，直接返回浓度场结果
        return x_phys, t_phys, cNO_pred, cOOH_pred
    else:
        # --- 开始传质速率的计算流程 ---
        # 1. 计算 D_NO
        De = get_de(rpm)
        D_NO = 2.21e-9 * De

        # 2. 准备边界 (x=0) 的输入数据
        # 我们复用之前计算的 t_phys，只需创建 x=0 的坐标
        x_at_boundary = np.zeros_like(t_phys) # shape: (nt, 1)

        # 3. 转换为Tensor，并为 x 坐标开启梯度追踪
        X_tensor_boundary = to_tensor(x_at_boundary)
        X_tensor_boundary.requires_grad_(True) # 关键！

        # 其他参数张量也需要对应边界的维度 (nt, 1)
        T_tensor_boundary = to_tensor(t_phys)
        pH_tensor_boundary = torch.full_like(T_tensor_boundary, pH)
        ch_tensor_boundary = torch.full_like(T_tensor_boundary, ch)
        rpm_tensor_boundary = torch.full_like(T_tensor_boundary, rpm)
        Q_tensor_boundary = torch.full_like(T_tensor_boundary, Q)

        # 4. 归一化并组合成模型输入
        x_star_boundary = X_tensor_boundary / (d_tensor / 2)
        t_star_boundary = T_tensor_boundary / ts_tensor

        X_input_boundary = torch.cat([
            x_star_boundary, t_star_boundary, pH_tensor_boundary,
            ch_tensor_boundary, rpm_tensor_boundary, Q_tensor_boundary
        ], dim=1)

        # 5. 执行带梯度追踪的前向传播
        # **注意**: 这一步绝对不能在 `with torch.no_grad()` 内部！
        cNO_boundary, _ = model(X_input_boundary)

        # 6. 计算梯度 ∂c_NO / ∂x
        dcNO_dx = torch.autograd.grad(
            outputs=cNO_boundary,
            inputs=X_tensor_boundary,
            grad_outputs=torch.ones_like(cNO_boundary),
            create_graph=False
        )[0]

        # 7. 计算传质速率 R_NO
        R_NO_tensor = -D_NO * dcNO_dx
        R_NO_numpy = R_NO_tensor.cpu().detach().numpy()

        # 返回所有结果
        return x_phys, t_phys, cNO_pred, cOOH_pred, R_NO_numpy


def merged_transport_solver(model, nx, nt, cno, Hno, Pno, T, G_N2, h, y_NO_in, r, r2, r1, pH=12.0, ch=1.0, rpm=1200, Q=20e-3 / 3600):
    """
    整合后的传质求解流程 (空间和时间点数可独立控制)

    参数:
    model: PINN模型
    nx: 空间网格密度
    nt: 时间网格密度
    cno: 浓度归一化系数
    Hno: 亨利系数
    Pno: 气相分压
    T: 温度(K)
    G_N2: 氮气流量
    h: 高度
    y_NO_in: 入口浓度
    r, r2, r1: 设备几何参数
    pH, ch, rpm, Q: 固定工况参数

    返回:
    dict: 包含所有传质参数 (这里简化为返回 k, ky_cal, y_out, y_no)
    """
    # Step 1: 生成预测数据 - 使用 nx 和 nt
    x_phys, t_phys, cNO_pred, cOOH_pred, R_NO = predict(model, nx, nt, pH, ch, rpm, Q, r, r2, r1, calculate_rate=True)
    # cNO_pred 的 shape 是 (nt, nx)

    # Step 2: 计算特征参数
    d = get_d(rpm, Q, r)
    ts = get_ts(rpm, Q, r, r2, r1)
    D1 = 2.21e-9 * get_de(rpm)
    holdup = get_holdup(rpm, Q, r, v0=1.0e-6, g0=100)
    a_bi = 6.0 * holdup / d

    # Step 3: 计算累积传质通量 (向量化优化)
    # cNO_pred shape is (nt, nx)
    # dx 是空间步长. x_phys = np.linspace(0, d / 2, nx)
    # 时间步长，保持原来的定义
    dt = ts / (nt - 1) if nt > 1 else ts

    # R_NO: shape (nt, 1)，已经是 -D * dc/dx at x=0(+)
    flux_t = R_NO.squeeze(-1)           # shape: (nt,)
    k = np.trapz(flux_t, dx=dt)              # 近似 ∫_0^{ts} J(t) dt

    # Step 4: 计算传质系数
    KL = (k) / (ts * cno)
    kg1 = 0.007954  # 预设气相传质系数
    # 防止 KL 为零导致除零错误
    Kg = 1 / (1 / kg1 + Hno / KL)
    ky_cal = 0.082 * T * (Pno * Kg)
    vol = pi * (r2 ** 2 - r1 ** 2) * h

    # Step 5: 求解出口浓度
    a1 = (1 - y_NO_in) / y_NO_in  # 预测出口系数1
    a2 = a_bi * pi * h * (r2 ** 2 - r1 ** 2) / G_N2  # 预测出口系数2

    # 防止指数溢出
    exponent = a2 * ky_cal
    if np.isscalar(exponent):
        # 如果是标量
        if exponent > 700:  # exp(709)左右会溢出
            y_out = 0.0  # 当指数非常大时，y_out趋近于0
        else:
            y_out = 1 / (a1 * np.exp(exponent) + 1)
    else:
        # 如果是数组
        safe_exp = np.where(exponent > 700, 0, np.exp(exponent))
        y_out = 1 / (a1 * safe_exp + 1)
        # 对于exponent>700的情况，设置y_out为0
        y_out = np.where(exponent > 700, 0.0, y_out)

    # Step 6: 计算去除率
    y_no = (1 - y_out / y_NO_in)

    return k, ky_cal, y_out, y_no










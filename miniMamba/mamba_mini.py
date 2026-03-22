import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _npo2(n):
    return 2 ** math.ceil(math.log2(n))


def _pad_npo2(x):
    return F.pad(x, (0, 0, 0, 0, 0, _npo2(x.size(1)) - x.size(1)))


class PScan(torch.autograd.Function):
    @staticmethod
    def _scan(A, X):
        B, D, L, _ = A.size()
        n, Aa, Xa = int(math.log2(L)), A, X
        for _ in range(n - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa, Xa = Aa[:, :, :, 1], Xa[:, :, :, 1]
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return
        Aa = A[:, :, 2**(n-2)-1:L:2**(n-2)]
        Xa = X[:, :, 2**(n-2)-1:L:2**(n-2)]
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])
        for k in range(n - 3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def _scan_rev(A, X):
        B, D, L, _ = A.size()
        n, Aa, Xa = int(math.log2(L)), A, X
        for _ in range(n - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])
            Aa, Xa = Aa[:, :, :, 0], Xa[:, :, :, 0]
        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return
        Aa = A[:, :, 0:L:2**(n-2)]
        Xa = X[:, :, 0:L:2**(n-2)]
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])
        for k in range(n - 3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        L = X_in.size(1)
        A = A_in.clone() if L == _npo2(L) else _pad_npo2(A_in)
        X = X_in.clone() if L == _npo2(L) else _pad_npo2(X_in)
        A, X = A.transpose(2, 1), X.transpose(2, 1)
        PScan._scan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, dY):
        A_in, X = ctx.saved_tensors
        L = dY.size(1)
        g  = dY.clone()  if L == _npo2(L) else _pad_npo2(dY)
        Ai = A_in        if L == _npo2(L) else _pad_npo2(A_in)
        g, Ai = g.transpose(2, 1), Ai.transpose(2, 1)
        PScan._scan_rev(F.pad(Ai[:, :, 1:], (0, 0, 0, 1)), g)
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * g[:, :, 1:])
        return Q.transpose(2, 1)[:, :L], g.transpose(2, 1)[:, :L]


pscan = PScan.apply


@dataclass
class MambaConfig:
    d_model:   int
    n_layers:  int
    dt_rank:   Union[int, str] = "auto"
    d_state:   int   = 16
    expand:    int   = 2
    d_conv:    int   = 4
    dt_min:    float = 0.001
    dt_max:    float = 0.1
    dt_init:   str   = "random"
    dt_scale:  float = 1.0
    dt_floor:  float = 1e-4
    norm_eps:  float = 1e-5
    bias:      bool  = False
    conv_bias: bool  = True
    use_pscan: bool  = True

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d, ed, r, n = cfg.d_model, cfg.d_inner, cfg.dt_rank, cfg.d_state

        self.in_proj  = nn.Linear(d, 2 * ed, bias=cfg.bias)
        self.conv1d   = nn.Conv1d(ed, ed, cfg.d_conv, groups=ed, padding=cfg.d_conv - 1, bias=cfg.conv_bias)
        self.x_proj   = nn.Linear(ed, r + 2 * n, bias=False)
        self.dt_proj  = nn.Linear(r, ed, bias=True)
        self.out_proj = nn.Linear(ed, d, bias=cfg.bias)

        std = r ** -0.5 * cfg.dt_scale
        if cfg.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -std, std)

        dt = torch.exp(
            torch.rand(ed) * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, n + 1, dtype=torch.float32).repeat(ed, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(ed))
        self.D._no_weight_decay = True

    def forward(self, x):
        _, L, _ = x.shape
        xr, z = self.in_proj(x).chunk(2, dim=-1)
        xr = F.silu(self.conv1d(xr.transpose(1, 2))[:, :, :L].transpose(1, 2))
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        delta, B, C = torch.split(
            self.x_proj(xr), [self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        dA = torch.exp(delta.unsqueeze(-1) * A)
        BX = delta.unsqueeze(-1) * B.unsqueeze(2) * xr.unsqueeze(-1)
        hs = pscan(dA, BX) if self.cfg.use_pscan else self._seq(dA, BX)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) + D * xr
        return self.out_proj(y * F.silu(z))

    def _seq(self, dA, BX):
        h = torch.zeros(BX.size(0), self.cfg.d_inner, self.cfg.d_state, device=dA.device)
        hs = []
        for t in range(BX.size(1)):
            h = dA[:, t] * h + BX[:, t]
            hs.append(h)
        return torch.stack(hs, dim=1)

    def step(self, x, cache):
        h, buf = cache
        xr, z = self.in_proj(x).chunk(2, dim=1)
        xn = xr.unsqueeze(2)
        xr = F.silu(self.conv1d(torch.cat([buf, xn], dim=2))[:, :, self.cfg.d_conv - 1])
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        delta, B, C = torch.split(
            self.x_proj(xr), [self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        dA = torch.exp(delta.unsqueeze(-1) * A)
        BX = delta.unsqueeze(-1) * B.unsqueeze(1) * xr.unsqueeze(-1)
        h = dA * (torch.zeros_like(dA) if h is None else h) + BX
        y = (h @ C.unsqueeze(-1)).squeeze(2) + D * xr
        return self.out_proj(y * F.silu(z)), (h, torch.cat([buf[:, :, 1:], xn], dim=2))


class ResidualBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(cfg)
        self.norm  = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x

    def step(self, x, cache):
        y, cache = self.mixer.step(self.norm(x), cache)
        return y + x, cache


class Mamba(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.layers = nn.ModuleList([ResidualBlock(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches

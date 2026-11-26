"""Frozen train-fitted condition prototypes and dynamic low-rank operators."""
import torch
from torch import nn
from torch.nn import functional as F


def entmax15(logits):
    """Exact alpha=1.5 entmax, evaluated in float32 for stable AMP routing."""
    x = logits.float() / 2
    x = x - x.max(-1, keepdim=True).values
    ordered = x.sort(dim=-1, descending=True).values
    rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    mean = ordered.cumsum(-1) / rho
    mean2 = ordered.square().cumsum(-1) / rho
    delta = (1 - rho * (mean2 - mean.square())) / rho
    tau = mean - delta.clamp_min(0).sqrt()
    support = (tau <= ordered).sum(-1, keepdim=True).clamp_min(1)
    threshold = tau.gather(-1, support - 1)
    return (x - threshold).clamp_min(0).square()


class ConditionPrototypeBank(nn.Module):
    def __init__(self, descriptor_dim, key_dim, count, temperature):
        super().__init__()
        self.temperature = temperature
        self.register_buffer('mean', torch.zeros(descriptor_dim))
        self.register_buffer('scale', torch.ones(descriptor_dim))
        self.register_buffer('projection', torch.zeros(descriptor_dim, key_dim))
        self.register_buffer('keys', torch.zeros(count, key_dim))
        self.register_buffer('ready', torch.tensor(False))

    def query(self, descriptors):
        return F.normalize(((descriptors.float() - self.mean) / self.scale) @ self.projection, dim=-1)

    def forward(self, descriptors):
        if not bool(self.ready):
            raise RuntimeError('Fit prototype conditions on training data first')
        logits = self.query(descriptors) @ self.keys.T / self.temperature
        return entmax15(logits)

    @torch.no_grad()
    def fit(self, descriptors, iterations, seed):
        x = descriptors.float().to(self.mean.device)
        k, d = self.keys.shape
        if len(x) < max(k, d) or d > x.shape[-1]:
            raise ValueError('Not enough samples/dimensions for PCA and prototypes')
        self.mean.copy_(x.mean(0))
        self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-4))
        standardized = (x - self.mean) / self.scale
        # Full SVD is deterministic here; never fit using validation/test descriptors.
        _, _, vh = torch.linalg.svd(standardized, full_matrices=False)
        self.projection.copy_(vh[:d].T)
        q = self.query(x)
        rng = torch.Generator(device=q.device).manual_seed(seed)
        centres = q[torch.randperm(len(q), generator=rng, device=q.device)[:k]].clone()
        for _ in range(iterations):
            similarity = q @ centres.T
            assignments = similarity.argmax(-1)
            sums = torch.zeros_like(centres).index_add_(0, assignments, q)
            counts = torch.bincount(assignments, minlength=k)
            empty = counts == 0
            sums[empty] = q[similarity.max(-1).values.argsort()[:int(empty.sum())]]
            centres = F.normalize(sums, dim=-1)
        self.keys.copy_(centres)
        self.ready.fill_(True)
        return (q @ centres.T).argmax(0).cpu().tolist()


class PrototypeDynamicsAdapter(nn.Module):
    def __init__(self, dim, leads, rank, count):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.lead = nn.Parameter(torch.zeros(leads, rank))
        self.up = nn.Parameter(torch.zeros(count, rank, dim))

    def forward(self, h, route):
        low = F.silu(self.down(self.norm(h)) + self.lead[None, :, None])
        # Mix operators once per trajectory; avoid materializing [B,T,N,K,D].
        operator = torch.einsum('bnk,krd->bnrd', route.to(low.dtype), self.up)
        return torch.einsum('btnr,bnrd->btnd', low, operator)



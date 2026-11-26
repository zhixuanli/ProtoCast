"""V1.5.5 condition bank that follows a backbone trained from scratch."""
import torch
from torch.nn import functional as F

from protocast.modules.prototype import ConditionPrototypeBank


def spherical_kmeans(q, centres, iterations):
    """Same assignment/update rule as V1.5.2, starting from given centres."""
    for _ in range(iterations):
        similarity = q @ centres.T
        assignments = similarity.argmax(-1)
        sums = torch.zeros_like(centres).index_add_(0, assignments, q)
        counts = torch.bincount(assignments, minlength=len(centres))
        empty = counts == 0
        sums[empty] = q[similarity.max(-1).values.argsort()[:int(empty.sum())]]
        centres = F.normalize(sums, dim=-1)
    return centres


class TrackedConditionPrototypeBank(ConditionPrototypeBank):
    """Standardization, PCA and keys are refitted as the shared features evolve.

    Refits always use the same training trajectories. The first fit is exactly
    V1.5.2's; later refits seed slot k with the new queries of the trajectories
    slot k owned last time, so slot k (and its operator B_k) keeps referring to
    the same group of training samples even when the feature basis rotates.
    """

    def __init__(self, descriptor_dim, key_dim, count, temperature, samples):
        super().__init__(descriptor_dim, key_dim, count, temperature)
        self.register_buffer('owners', torch.full((samples,), -1, dtype=torch.long))

    @torch.no_grad()
    def refit(self, descriptors, iterations, seed):
        x = descriptors.float().to(self.mean.device)
        n, (k, d) = len(x), self.keys.shape
        if n > len(self.owners):
            raise ValueError('More refit samples than the bank tracks')
        previous = self.owners[:n].clone()
        tracked = bool(self.ready) and bool((previous >= 0).all())
        if not tracked:
            self.fit(x, iterations, seed)
        else:
            self.mean.copy_(x.mean(0))
            self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-4))
            _, _, vh = torch.linalg.svd((x - self.mean) / self.scale, full_matrices=False)
            self.projection.copy_(vh[:d].T)
            q = self.query(x)
            sums = torch.zeros_like(self.keys).index_add_(0, previous, q)
            empty = torch.bincount(previous, minlength=k) == 0
            if empty.any():
                # A slot that owned nothing is re-seeded where the others fit worst.
                worst = (q @ F.normalize(sums[~empty], dim=-1).T).max(-1).values.argsort()
                sums[empty] = q[worst[:int(empty.sum())]]
            self.keys.copy_(spherical_kmeans(q, F.normalize(sums, dim=-1), iterations))
        current = (self.query(x) @ self.keys.T).argmax(-1)
        agreement = float((current == previous).float().mean()) if tracked else None
        self.owners.fill_(-1)
        self.owners[:n] = current
        return dict(slot_agreement=agreement, **self.diagnostics(x))


    @torch.no_grad()
    def diagnostics(self, descriptors):
        q = self.query(descriptors.float().to(self.mean.device))
        counts = torch.bincount((q @ self.keys.T).argmax(-1), minlength=len(self.keys)).float()
        usage = counts / counts.sum()
        cosine = self.keys @ self.keys.T
        cosine.fill_diagonal_(-1)
        return dict(top1_usage_perplexity=float((-(usage * usage.clamp_min(1e-12).log()).sum()).exp()),
                    unused_prototypes=int((counts == 0).sum()),
                    key_max_pairwise_cosine=float(cosine.max()))

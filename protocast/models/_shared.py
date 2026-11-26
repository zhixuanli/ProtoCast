"""Internal grouped backbone and prototype state used by ProtoCast."""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from protocast.modules.blocks import GatedBlock, patchify, sinusoidal, unpatchify
from protocast.modules.prototype import PrototypeDynamicsAdapter
from protocast.modules.tracking import TrackedConditionPrototypeBank
from protocast.utils.checkpoint import seed_all


_ATTENTION_BATCH_CHUNK = 4096


def _apply_independent_batches(module, x):
    """Bound CUDA attention launch size without splitting any attention axis.

    The first dimension holds independent sequences (patch trajectories or
    four-variable groups), so concatenation preserves their order and gradients.
    Chunking changes dropout RNG order during training, not its distribution or
    the model's time/space/variable attention domains or optimizer batch size.
    """
    if x.shape[0] <= _ATTENTION_BATCH_CHUNK:
        return module(x)
    return torch.cat([module(chunk) for chunk in x.split(_ATTENTION_BATCH_CHUNK, dim=0)], dim=0)


class GroupedVariableBackbone(nn.Module):
    """Keep one latent per variable while sharing temporal and spatial weights.

    Each variable owns its patch encoder and decoder. The variable-attention
    block attends across the four groups at each forecast lead and patch;
    temporal/spatial blocks retain their original time/space attention axes.
    """
    def __init__(self, config, variable_groups):
        super().__init__()
        self.config = dict(config)
        self.h, self.w, self.p = config['height'], config['width'], config['patch_size']
        self.tin, self.tout = config['input_steps'], config['output_steps']
        self.channels, self.dim = config['channels'], config['dim']
        self.target_indices = config.get('target_indices', list(range(self.channels)))
        self.variable_groups = {name: tuple(indices) for name, indices in variable_groups.items()}
        covered = [channel for indices in self.variable_groups.values() for channel in indices]
        if (len(self.variable_groups) != 4 or any(not indices for indices in self.variable_groups.values())
                or sorted(covered) != list(range(self.channels))):
            raise ValueError('Four variable groups must cover every input/output channel exactly once')
        if self.target_indices != list(range(self.channels)):
            raise ValueError('ProtoCast expects all variables as forecast targets in their original order')
        if self.h % self.p or self.w % self.p or self.dim % config['heads'] or config['depth'] < 1:
            raise ValueError('Grid must divide into patches; dim must divide by heads; depth must be positive')
        self.encoders = nn.ModuleDict({name: nn.Linear(len(indices) * self.p ** 2, self.dim)
                                       for name, indices in self.variable_groups.items()})
        self.decoders = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(self.dim),
                                        nn.Linear(self.dim, len(indices) * self.p ** 2))
                                       for name, indices in self.variable_groups.items()})
        n = self.h // self.p * (self.w // self.p)
        self.register_buffer('time_position', sinusoidal(self.tin, self.dim)[None, :, None, None])
        position = (sinusoidal(self.h // self.p, self.dim)[:, None]
                    + sinusoidal(self.w // self.p, self.dim)[None, :]).reshape(n, self.dim)
        self.register_buffer('space_position', position[None, None, None])
        args = (self.dim, config['heads'], config['dropout'])
        self.temporal = nn.Sequential(*[GatedBlock(*args) for _ in range(config['depth'])])
        self.time_projection = nn.Linear(self.tin, self.tout)
        self.spatial = nn.Sequential(*[GatedBlock(*args) for _ in range(config['depth'])])
        self.variable_attention = GatedBlock(*args)

    def prefix(self, x):
        """Return [B,Tout,G,N,D] and local-history/fused-context descriptors."""
        if tuple(x.shape[1:]) != (self.tin, self.channels, self.h, self.w):
            raise ValueError(f'Unexpected input shape: {tuple(x.shape)}')
        groups = [self.encoders[name](patchify(x[:, :, indices], self.p))
                  for name, indices in self.variable_groups.items()]
        z = torch.stack(groups, dim=2) + self.time_position + self.space_position
        b, t, g, n, d = z.shape
        # The fused temporal SDPA path produced nonfinite gradients on a real
        # H200 training batch, also in FP32. Math SDPA keeps this short time
        # axis stable; spatial/variable attention retain their normal backends.
        # Equations and dropout probability are unchanged, but dropout RNG
        # consumption can differ from the fused kernel when resuming a run.
        with sdpa_kernel(SDPBackend.MATH):
            temporal = _apply_independent_batches(self.temporal,
                                                  z.permute(0, 2, 3, 1, 4).reshape(b * g * n, t, d))
        # This history serves fixed-key routing only; reuse the forecast's
        # temporal features without retaining an extra descriptor graph.
        with torch.no_grad():
            history = torch.cat((temporal.mean(1), temporal[:, -1] - temporal[:, 0]), -1)
            history = history.reshape(b, g, n, 2 * d)
        z = self.time_projection(temporal.transpose(1, 2)).transpose(1, 2)
        z = z.reshape(b, g, n, self.tout, d).permute(0, 3, 1, 2, 4)
        z = self.spatial[:-1](z.reshape(b * self.tout * g, n, d)).reshape(b, self.tout, g, n, d)
        fused = _apply_independent_batches(self.variable_attention,
                                           z.permute(0, 1, 3, 2, 4).reshape(b * self.tout * n, g, d))
        h = fused.reshape(b, self.tout, n, g, d).permute(0, 1, 3, 2, 4)
        context = h.mean(1)
        descriptors = {name: torch.cat((history[:, index], context[:, index]), -1)
                       for index, name in enumerate(self.variable_groups)}
        return h, descriptors

    def decode(self, h, x):
        """Decode each adapted variable without another cross-variable merge."""
        b, t, g, n, d = h.shape
        z = self.spatial[-1](h.reshape(b * t * g, n, d)).reshape(b, t, g, n, d)
        channels = [None] * self.channels
        for index, (name, indices) in enumerate(self.variable_groups.items()):
            forecast = unpatchify(self.decoders[name](z[:, :, index]), self.p, self.h, self.w)
            for local_channel, channel in enumerate(indices):
                channels[channel] = forecast[:, :, local_channel:local_channel + 1]
        return torch.cat(channels, dim=2) + x[:, -1:]

    def forward(self, x):
        return self.decode(self.prefix(x)[0], x)



class _GroupedPrototypeModel(nn.Module):
    """Independent variable prototype banks on four fused, distinct latents."""
    variant = 'entmax'

    def __init__(self, backbone, settings):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.variable_groups = dict(backbone.variable_groups)
        d, t = backbone.dim, backbone.tout
        self.banks = nn.ModuleDict({
            name: TrackedConditionPrototypeBank(3 * d, settings['key_dim'], settings['num_prototypes'],
                                                settings['temperature'], settings['init_samples'])
            for name in self.variable_groups
        })
        self.adapters = nn.ModuleDict({
            name: PrototypeDynamicsAdapter(d, t, settings['rank'], settings['num_prototypes'])
            for name in self.variable_groups
        })

    def prefix(self, x):
        return self.backbone.prefix(x)

    @torch.no_grad()
    def refit_banks(self, descriptors, iterations, seed):
        if set(descriptors) != set(self.variable_groups):
            raise ValueError('Descriptors must provide every variable group exactly once')
        return {name: self.banks[name].refit(descriptors[name], iterations, seed + index)
                for index, name in enumerate(self.variable_groups)}





